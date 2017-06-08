#include "../Audacity.h"
#include "AudioAlign.h"

#include <math.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <wx/dir.h>
#include <wx/file.h>
#include <wx/filefn.h>
#include <wx/intl.h>

#include "../FFT.h"
#include "../LabelTrack.h"
#include "../Prefs.h"
#include "../Project.h"
#include "../Resample.h"
#include "../SampleFormat.h"
#include "../ShuttleGui.h"
#include "../Tags.h"
#include "../WaveTrack.h"
#include "../import/Import.h"
#include "../widgets/valnum.h"
#include "TimeWarper.h"

BEGIN_EVENT_TABLE(EffectAudioAlign, wxEvtHandler)
END_EVENT_TABLE()

namespace {

// Summary of an audio track. There is one value for each time window
// of length 'time_per_summarized_sample_millis'.
struct AudioSummary {
  // TODO: rely on same parameter in options?
  double time_per_summarized_sample_millis = 0;
  // One element for each time window, summarized according to
  // AudioAlignOptions.summary_type.
  std::vector<float> summarized_amplitudes;
  // Returns a windowed summary [window_start, window_end).
  AudioSummary Window(int window_start, int window_end) const {
    wxASSERT(window_start >= 0);
    wxASSERT(window_end >= 0);
    wxASSERT(window_start <= window_end);
    wxASSERT(window_start <= summarized_amplitudes.size());
    wxASSERT(window_end <= summarized_amplitudes.size());
    AudioSummary ret;
    ret.time_per_summarized_sample_millis = time_per_summarized_sample_millis;
    std::copy(summarized_amplitudes.begin() + window_start,
              summarized_amplitudes.begin() + window_end,
              std::back_inserter(ret.summarized_amplitudes));
    return ret;
  }
};

// Inspired from libscoralign/regression.h, with automatic removal of
// outliers for more robustness.
class RobustRegression {
 public:
  void point(float x, float y) {
    xs.push_back(x);
    ys.push_back(y);
  }
  void regress() {
    std::vector<double> weights(xs.size(), 1);
    const int MAX_ITER = 100;
    for (int k = 0; k < MAX_ITER; ++k) {
      std::vector<double> residuals;
      RegressOnce(weights, &residuals, &a, &b);
      std::vector<double> r = residuals;
      auto middle = r.begin() + (r.size() / 2);
      std::nth_element(r.begin(), middle,
                       r.end());  // todo: replace with nth_element.
      double median = *middle;
      if (median < 1e-10) break;
      for (int i = 0; i < weights.size(); ++i) {
        // Simple outlier handling: remove points whose residual is
        // greater than median_residual * 5.
        weights[i] = residuals[i] < median * 5;
      }
    }
  }
  float f(float x) { return a + b * x; }

 private:
  void RegressOnce(const std::vector<double>& weights,
                   std::vector<double>* residuals, double* aa,
                   double* bb) const {
    double sumxx = 0;  // sum of x^2
    double sumyy = 0;  // sum of y^2
    double sumxy = 0;  // sum of xy
    double sumx = 0;   // sum of x
    double sumy = 0;   // sum of y
    double n = 0;      // sum of point weights.

    for (int i = 0; i < xs.size(); ++i) {
      double w = weights[i];
      double x = xs[i] * w;
      double y = ys[i] * w;
      sumx += x;
      sumy += y;
      sumxx += x * x;
      sumyy += y * y;
      sumxy += x * y;
      n += weights[i];
    }
    double sxx = sumxx - sumx * sumx / n;
    double sxy = sumxy - sumx * sumy / n;
    *bb = sxy / sxx;
    *aa = (sumy - *bb * sumx) / n;
    for (int i = 0; i < xs.size(); ++i) {
      residuals->push_back(fabs(*aa + *bb * xs[i] - ys[i]));
    }
  }
  double a, b;                 // regression line is a + b*x
  std::vector<double> xs, ys;  // input points
};

int RoundNextPowerOf2(int s) {
  if (s == 0) return 0;
  int power = 1;
  while (power < s) {
    power *= 2;
  }
  return power;
}

// Helper function to print a vector either in CSV format, or with R
// syntax.
template <typename T>
void PrintVector(const std::vector<T>& v, const std::string& name,
                 std::ostream& out, bool csv_format = true) {
  if (csv_format) {
    for (int i = 0; i < v.size(); ++i) {
      out << v[i] << "\n";
    }
    out << std::endl;
  } else {
    out << name << "= c(";
    for (int i = 0; i < v.size(); ++i) {
      if (i > 0) out << ",";
      out << v[i];
    }
    out << ")" << std::endl;
  }
}

double SumAbsBuffer(const SampleBuffer& buffer, int len) {
  double ret = 0;
  for (int i = 0; i < len; ++i) {
    ret += fabs(((float*)buffer.ptr())[i]);
  }
  return ret;
}

double MaxBuffer(const SampleBuffer& buffer, int len) {
  double ret = 0;
  for (int i = 0; i < len; ++i) {
    ret = std::max<double>(ret, fabs(((float*)buffer.ptr())[i]));
  }
  return ret;
}

// Returns the number of zero crossings in the buffer.
// See SummaryType::kZeroCrossings for more details.
double ZeroCrossingsBuffer(const SampleBuffer& buffer, int len) {
  double ret = 0;
  for (int i = 1 /*on purpose*/; i < len; ++i) {
    if (((float*)buffer.ptr())[i - 1] * ((float*)buffer.ptr())[i] < 0) {
      ++ret;
    }
  }
  return ret;
}

// See SummaryType::kSpectralFlatness for more details.
double SpectralFlatnessBuffer(const SampleBuffer& buffer, int len) {
  int rounded_len = RoundNextPowerOf2(len);
  std::vector<float> power_spectrum(rounded_len / 2 + 1, 0);
  std::vector<float> input(rounded_len, 0);
  for (int i = 0; i < len; ++i) {
    input[i] = ((float*)buffer.ptr())[i];
  }
  PowerSpectrum(rounded_len, &input[0], &power_spectrum[0]);
  double sum_logs = 0;
  double sum = 0;
  for (auto v : power_spectrum) {
    sum_logs += std::log(v);
    sum += v;
  }
  if (sum <= 0) return 0;
  return std::exp(sum_logs / power_spectrum.size()) / sum;
}

// Summarizes a track.
// Also prints the summary on "out_file", for debugging purposes.
AudioSummary SummarizeTrack(const WaveTrack* track,
                            const AudioAlignOptions& options,
                            std::ofstream& out_file) {
  std::chrono::time_point<std::chrono::system_clock> time_start =
      std::chrono::system_clock::now();
  AudioSummary ret;
  ret.time_per_summarized_sample_millis =
      options.time_per_summarized_sample_millis;
  const int buff_size =
      track->GetRate() * ret.time_per_summarized_sample_millis / 1000 + 2;
  double t = track->GetStartTime();
  SampleBuffer buffer(buff_size, floatSample);
  while (t < track->GetEndTime()) {
    const double next_t = t + ret.time_per_summarized_sample_millis / 1000;
    sampleCount buffer_start = track->TimeToLongSamples(t);
    sampleCount buffer_end = track->TimeToLongSamples(next_t);
    size_t len = (buffer_end - buffer_start).as_size_t();
    wxASSERT(len <= buff_size);
    track->Get(buffer.ptr(), floatSample, buffer_start, len);
    double summary_value = 0;
    switch (options.summary_type) {
      case SummaryType::kSumAbs:
        summary_value = SumAbsBuffer(buffer, len);
        break;
      case SummaryType::kMax:
        summary_value = MaxBuffer(buffer, len);
        break;
      case SummaryType::kZeroCrossings:
        summary_value = ZeroCrossingsBuffer(buffer, len);
        break;
      case SummaryType::kSpectralFlatness:
        summary_value = SpectralFlatnessBuffer(buffer, len);
        break;
    }
    ret.summarized_amplitudes.push_back(summary_value);
    t = next_t;
  }
  PrintVector(ret.summarized_amplitudes,
              std::string("summarized_") + track->GetName().ToStdString(),
              out_file);

  int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now() - time_start)
                       .count();
  std::cout << "Summarizing " << track->GetName() << " took " << elapsed_ms
            << "ms" << std::endl;
  return ret;
}

double CorrelationNormalization(double match_fraction) { return 1; }

// Simple class to represent an interval [begin, end).
class Interval {
 public:
  Interval(double begin, double end) : begin_(begin), end_(end) {}

  // Translates the interval to the right.
  void TranslateRight(double translation) {
    begin_ += translation;
    end_ += translation;
  }
  double Size() { return std::max(0., end_ - begin_); }
  // Intersects this interval with 'other', in place.
  void Intersect(const Interval& other) {
    begin_ = std::max(begin_, other.begin_);
    end_ = std::min(end_, other.end_);
  }

 private:
  double begin_;
  double end_;
};

float GetModuloSize(const std::vector<float>& v, int i) {
  i %= v.size();
  if (i < 0) i += v.size();
  wxASSERT(i >= 0);
  wxASSERT(i < v.size());
  return v[i];
}

bool IsWithinMatchPercentage(int match_position, int min_match_percent,
                             int reference_size, int other_size,
                             int total_size) {
  const Interval interval_ref(0, reference_size);
  Interval interval_other(0, other_size);
  if (match_position <= reference_size) {
    interval_other.TranslateRight(match_position);
  } else if (match_position >= total_size - other_size) {
    interval_other.TranslateRight(match_position - total_size);
  } else {
    return false;
  }
  interval_other.Intersect(interval_ref);
  return interval_other.Size() / std::min(reference_size, other_size) * 100 >=
         min_match_percent;
}

// Finds the best alignment shift in the given cross correlation.
int FindShift(const std::vector<float>& normalized_cross_correlation,
              const AudioAlignOptions& options, int reference_size,
              int other_size, bool print_debug = true) {
  switch (options.shift_finding_method) {
    case ShiftFindingMethod::kMax: {
      auto max_it = std::max_element(normalized_cross_correlation.begin(),
                                     normalized_cross_correlation.end());
      return max_it - normalized_cross_correlation.begin();
    }
    case ShiftFindingMethod::kMaxPeakToAveragePowerRatio: {
      // We require at least this percentage of match of the smallest
      // fragment.  This is to avoid spurious matches of only a few
      // percents of tracks.
      const int kMinMatchPercent = 20;
      // Time window for the max-peak-to-average-power-ratio computation.
      const double kTimeWindowSec = 1;
      const int kWindowHalfSize =
          kTimeWindowSec / options.time_per_summarized_sample_millis * 1000 / 2;
      double sum_squares = 0;
      std::vector<float> sum_squares_debug;
      for (int i = -kWindowHalfSize - 1; i < kWindowHalfSize; ++i) {
        sum_squares +=
            std::pow(GetModuloSize(normalized_cross_correlation, i), 2);
      }
      // Peak to average power ratios.
      std::vector<float> papr(normalized_cross_correlation.size());
      for (int i = -kWindowHalfSize, j = 0, k = kWindowHalfSize;
           j < normalized_cross_correlation.size(); i++, j++, k++) {
        // Remove [i-1], add [k]
        sum_squares -=
            std::pow(GetModuloSize(normalized_cross_correlation, i - 1), 2);
        sum_squares +=
            std::pow(GetModuloSize(normalized_cross_correlation, k), 2);

        double sum_squares_naive = 0;
        for (int m = i; m <= k; ++m) {
          sum_squares_naive +=
              std::pow(GetModuloSize(normalized_cross_correlation, m), 2);
        }
        // TODO: compare sum_squares and sum_squares_naive, and check
        // for numerical drift.
        sum_squares_debug.push_back(sum_squares);
        if (sum_squares <= 0 ||
            !IsWithinMatchPercentage(j, kMinMatchPercent, reference_size,
                                     other_size,
                                     normalized_cross_correlation.size())) {
          papr[j] = 0;
        } else {
          papr[j] = std::pow(normalized_cross_correlation[j], 2) / sum_squares *
                    // The goal of this is to also take into account the
                    // amplitude of the peaks, not just their peakiness.
                    std::pow(normalized_cross_correlation[j],
                             options.correlation_normalization_power);
        }
      }
      if (print_debug) {
        std::ofstream out_papr("papr.csv");
        PrintVector(papr, "papr", out_papr);
        std::ofstream out_sum_squares("sum_squares.csv");
        PrintVector(sum_squares_debug, "sum_squares", out_sum_squares);
      }
      auto max_it = std::max_element(papr.begin(), papr.end());
      return max_it - papr.begin();
    }
  }
  wxFAIL_MSG("Unknown shift finding method " +
             std::to_string(options.shift_finding_method));
}

// Aligns two tracks, given by their summaries, and returns the time shift.
// Namely, returns:
// (shift in number of summarized windows, time difference in seconds).
// If positive, it means 'to_align' should be shifted to the right by
// the corresponding value.
//
// The algorithm is to compute the cross-correlation function of the
// two track summaries, and deduce the best shift from this function
// using one ShiftFindingMethod. The cross-correlation function is
// computed in O(n log n) using FFT, thanks to the convolution
// theorem.
// corr(a, b) = ifft(fft(a_and_zeros) * conj(fft(b_and_zeros)))
// See Numerical Recipes in C, section 13-2 for details on convulution theorem:
// http://www.aip.de/groups/soe/local/numres/bookcpdf/c13-2.pdf
std::pair<int, double> Align(const AudioSummary& reference,
                             const AudioSummary& to_align,
                             AudioAlignOptions options,
                             bool print_debug = true) {
  std::chrono::time_point<std::chrono::system_clock> time_start =
      std::chrono::system_clock::now();
  // With the convolution theorem we get cross-correlation signals
  // assumed to be periodic. To make sure we don't get spurious
  // cross-correlations due to signals wrapping around, we need to
  // multiply by 2.
  int rounded_size =
      2 * RoundNextPowerOf2(std::max(reference.summarized_amplitudes.size(),
                                     to_align.summarized_amplitudes.size()));
  if (rounded_size == 0) {
    return std::make_pair(0, 0);
  }
  std::vector<float> ref(rounded_size, 0);
  std::copy(reference.summarized_amplitudes.begin(),
            reference.summarized_amplitudes.end(), ref.begin());

  std::vector<float> reference_fft_real(rounded_size, 0);
  std::vector<float> reference_fft_imag(rounded_size, 0);

  RealFFT(ref.size(), &ref[0], &reference_fft_real[0], &reference_fft_imag[0]);
  //  PrintVector(reference_fft_real, "reference_fft_real", std::cout, false);
  //  PrintVector(reference_fft_imag, "reference_fft_imag", std::cout, false);
  // reference fft is correct (checked in R).

  std::vector<float> other(rounded_size, 0);
  // Beware!! Reversing the input must be done like that to avoid
  // being always wrong by one offset.
  /*
  std::copy(to_align.summarized_amplitudes.begin() + 1,
            to_align.summarized_amplitudes.end(),
            other.rbegin());
  other[0] = to_align.summarized_amplitudes[0];
  */
  std::copy(to_align.summarized_amplitudes.begin(),
            to_align.summarized_amplitudes.end(), other.begin());

  std::vector<float> other_fft_real(rounded_size, 0);
  std::vector<float> other_fft_imag(rounded_size, 0);
  RealFFT(other.size(), &other[0], &other_fft_real[0], &other_fft_imag[0]);
  //  PrintVector(other_fft_real, "other_fft_real", std::cout, false);
  //  PrintVector(other_fft_imag, "other_fft_imag", std::cout, false);

  // (a + ib) * (x - iy) = ax + by + (-ay + bx) i.
  std::vector<float> multiplied_real(rounded_size, 0);
  std::vector<float> multiplied_imag(rounded_size, 0);
  for (int i = 0; i < rounded_size; ++i) {
    multiplied_real[i] = reference_fft_real[i] * other_fft_real[i] +
                         reference_fft_imag[i] * other_fft_imag[i];
    multiplied_imag[i] = -reference_fft_real[i] * other_fft_imag[i] +
                         reference_fft_imag[i] * other_fft_real[i];
  }

  //  PrintVector(multiplied_real, "multiplied_real", std::cout, false);
  //  PrintVector(multiplied_imag, "multiplied_imag", std::cout, false);

  std::vector<float> cross_correlation(rounded_size, 0);
  InverseRealFFT(rounded_size, &multiplied_real[0], &multiplied_imag[0],
                 &cross_correlation[0]);

  if (print_debug) {
    std::ofstream out_cross("cross.csv");
    PrintVector(cross_correlation, "cross", out_cross);
  }

  // TODO: normalization does nothing at the moment, disable it.
  std::vector<float> normalized_cross_correlation(cross_correlation.size());
  std::vector<float> normalization_factors(cross_correlation.size());
  const Interval interval_ref(0, reference.summarized_amplitudes.size());
  for (int i = 0; i < cross_correlation.size(); ++i) {
    Interval interval_other(0, to_align.summarized_amplitudes.size());
    if (i <= reference.summarized_amplitudes.size()) {
      interval_other.TranslateRight(i);
    }
    if (i >= rounded_size - to_align.summarized_amplitudes.size()) {
      interval_other.TranslateRight(i - rounded_size);
    }
    interval_other.Intersect(interval_ref);
    const double match_fraction =
        interval_other.Size() / std::min(reference.summarized_amplitudes.size(),
                                         to_align.summarized_amplitudes.size());
    normalized_cross_correlation[i] =
        cross_correlation[i] * CorrelationNormalization(match_fraction);
    normalization_factors[i] = CorrelationNormalization(match_fraction);
  }

  if (print_debug) {
    std::ofstream out_normalized_cross("normalized_cross.csv");
    PrintVector(normalized_cross_correlation, "normalized_cross",
                out_normalized_cross);
    std::ofstream out_normalized2_cross("normalized2_cross.csv");
    PrintVector(normalization_factors, "normalized2_cross",
                out_normalized2_cross);
    std::cout << "Computing cross correlations took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - time_start)
                     .count()
              << "ms" << std::endl;
  }

  const int raw_shift =
      FindShift(normalized_cross_correlation, options,
                reference.summarized_amplitudes.size(),
                to_align.summarized_amplitudes.size(), print_debug);
  int shift = raw_shift;
  // cross[t] = sum(reference[i + t] * to_align[i], i) =
  //     sum(reference[i] * to_align[i - t], i).
  // shift is in [0, rounded_size], signals are assumed to have period
  // rounded_size.
  // if shift < size(reference), then we are good (we translate
  // 'to_align' right by shift >= 0)
  // if shift > rounded_size - size(to_align), then we translate 'to_align'
  // by shift - rounded_size > 0.
  if (shift <= reference.summarized_amplitudes.size()) {
    // Nothing to do.
  } else if (shift >= rounded_size - to_align.summarized_amplitudes.size()) {
    shift -= rounded_size;
  } else {
    std::cout << "Invalid shoft: " << shift << " time shift s: "
              << (shift * reference.time_per_summarized_sample_millis / 1000)
              << " size: " << rounded_size << " "
              << reference.summarized_amplitudes.size() << " "
              << to_align.summarized_amplitudes.size() << std::endl;
    return std::make_pair(0, 0);
  }
  double time_shift_s =
      shift * reference.time_per_summarized_sample_millis / 1000;
  wxASSERT(to_align.time_per_summarized_sample_millis ==
           reference.time_per_summarized_sample_millis);
  if (print_debug) {
    std::cout << "raw shift: " << raw_shift << " Shift: " << shift
              << " time shift s: " << time_shift_s << " size: " << rounded_size
              << " " << reference.summarized_amplitudes.size() << " "
              << to_align.summarized_amplitudes.size() << std::endl;

    std::cout << "Aligning took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - time_start)
                     .count()
              << "ms" << std::endl;
  }

  return std::make_pair(shift, time_shift_s);
}

// Finds the time drift between two globally aligned summarized tracks.
// Returns (corrected_offset, speed_multiplier)
// 'speed multiplier' is the multiplier we should apply to 'to_align' so that
// it's correctly aligned.
// 'corrected_offset takes into account the input 'offset'. It's the
// amount that 'to_align' should be offset to the right in order to be
// properly aligned with 'reference'.
//
// Algorithm:
// 1. Splits both tracks into 60s chunks.
// 2. Aligns the corresponding chunks 1:1. We get one integer shift
//    for each chunk.
// 3. Runs a robust linear regression of the sequence of shifts, which
//    allows detecting a clock speed difference between the two tracks.
std::pair<double, double> FindTimeDrift(
    int offset,  // Initial offset from global alignment.
    const AudioSummary& reference, const AudioSummary& to_align,
    const AudioAlignOptions& options) {
  const int window_size = options.time_drift_detection_window_s * 1000 /
                          options.time_per_summarized_sample_millis;
  wxASSERT(window_size > 0);
  // Just to output to R, not used in computations.
  std::vector<int> offsets;
  std::vector<double> window_centers;

  RobustRegression regression;
  for (int window_start = 0, window_end = window_size;
       window_end <= to_align.summarized_amplitudes.size();
       window_end += window_size, window_start += window_size) {
    int reference_window_start = window_start + offset;
    if (reference_window_start < 0) {
      continue;
    }
    int reference_window_end = reference_window_start + window_size;
    if (reference_window_end > reference.summarized_amplitudes.size()) {
      continue;
    }
    AudioSummary to_align_windowed = to_align.Window(window_start, window_end);
    AudioSummary reference_windowed =
        reference.Window(reference_window_start, reference_window_end);
    int offset =
        Align(reference_windowed, to_align_windowed, options, false).first;
    // TODO: perform a robust linear regression. In the meantime,
    // exclude points that are most certainly outliers.
    // TODO: improve that, we are sometimes failing because of this.
    if (abs(offset) > 30) {
      continue;
    }
    offsets.push_back(offset);
    window_centers.push_back((reference_window_start + reference_window_end) /
                             2.);
    regression.point((window_start + window_end) / 2., offset);
  }
  if (offsets.size() < 5) {
    std::cout << "Not enough windows to detect clock drift: " << offsets.size()
              << " vs: " << 5 << std::endl;
    // Not enough length to detect clock drift.
    return std::make_pair(offset, 1);
  }
  std::ofstream out_offsets("offsets.csv");
  PrintVector(offsets, "", out_offsets);
  std::ofstream out_window_centers("window_centers.csv");
  PrintVector(window_centers, "", out_window_centers);

  regression.regress();
  double corrected_offset = offset + regression.f(0);
  double speed_multiplier = 1 / (1. + regression.f(1) - regression.f(0));
  std::cout << "corrected offset: " << corrected_offset
            << " raw regression offset: " << regression.f(0)
            << " input offset: " << offset
            << " speed multiplier: " << speed_multiplier << std::endl;
  return std::make_pair(corrected_offset, speed_multiplier);
}

// Runs the full alignment algorithm:
// - Summarized both tracks.
// - Aligns summaries globally with Align().
// - Finds clock differences with FindTimeDrift().
// Returns (time offset in second, time_multiplier).
std::pair<double, double> AlignTracks(const WaveTrack* reference_track,
                                      const WaveTrack* other_track,
                                      AudioAlignOptions options) {
  std::cout << "Summarizing reference track: " << reference_track->GetName()
            << " offset: " << reference_track->GetOffset()
            << " end time: " << reference_track->GetEndTime() << std::endl;
  std::ofstream out_ref("ref.csv");
  AudioSummary reference_summary =
      SummarizeTrack(reference_track, options, out_ref);
  std::ofstream out_other("other.csv");
  AudioSummary summary = SummarizeTrack(other_track, options, out_other);
  std::pair<int, double> offset = Align(reference_summary, summary, options);

  std::pair<double, double> ret;
  if (options.correct_time_drift) {
    std::pair<double, double> drift =
        FindTimeDrift(offset.first, reference_summary, summary, options);
    // convert from num window offset to seconds.
    ret.first = drift.first * options.time_per_summarized_sample_millis / 1000;
    ret.second = drift.second;
  } else {
    ret.first = offset.first;
    ret.second = 1;
  }

  std::ofstream out_file;
  out_file.open("audio.r");
  out_file << "ref = read.csv('ref.csv', header=F)" << std::endl;
  out_file << "other = read.csv('other.csv', header=F)" << std::endl;
  out_file << "cross = read.csv('cross.csv', header=F)" << std::endl;
  out_file << "plot(cross$V1, type='l')" << std::endl;
  out_file << "normalized_cross = read.csv('normalized_cross.csv', header=F)"
           << std::endl;
  out_file << "dev.new(); plot(normalized_cross$V1, type='l')" << std::endl;
  out_file << "normalized2_cross = read.csv('normalized2_cross.csv', header=F)"
           << std::endl;
  out_file << "dev.new(); plot(normalized2_cross$V1, type='l')" << std::endl;
  out_file << "papr = read.csv('papr.csv', header=F)" << std::endl;
  out_file << "dev.new(); plot(papr$V1, type='l')" << std::endl;
  out_file << "sum_squares = read.csv('sum_squares.csv', header=F)"
           << std::endl;
  out_file << "dev.new(); plot(sum_squares$V1, type='l')" << std::endl;
  out_file << "offsets = read.csv('offsets.csv', header=F)" << std::endl;
  out_file << "dev.new(); plot(offsets$V1, type='l')" << std::endl;
  out_file << "window_centers = read.csv('window_centers.csv', header=F)"
           << std::endl;
  out_file << "dev.new(); plot(window_centers$V1, type='l')" << std::endl;
  return ret;
}

// For the benchmark, computes the name of the transform from the filename.
// E.g. from:
// /home/raphael/audio_alignment/benchmark_out/orchestra_queen_park_diff_devices_diff_sampling_rates/volume/full/a.flac
// returns 'volume:full'.
std::string GetTransformFromFilename(const std::string& filename) {
  std::string dir = filename.substr(0, filename.rfind('/'));
  std::string time_part = dir.substr(dir.rfind('/') + 1);
  std::string dir_without_time_part = dir.substr(0, dir.rfind('/'));
  std::string transform_part =
      dir_without_time_part.substr(dir_without_time_part.rfind('/') + 1);
  return transform_part + ":" +
         (time_part == "full" ? time_part : std::string("60s"));
}

std::string PrintMap(const std::map<std::string, int>& m) {
  std::string s;
  for (const auto& e : m) {
    if (!s.empty()) s += ",";
    s += e.first + ":" + std::to_string(e.second);
  }
  return s;
}

}  // namespace

EffectAudioAlign::EffectAudioAlign(bool bench) { m_Bench = bench; }

EffectAudioAlign::~EffectAudioAlign() {}

wxString EffectAudioAlign::GetSymbol() {
  return m_Bench ? AUDIOALIGN_BENCHMARK_PLUGIN_SYMBOL
                 : AUDIOALIGN_PLUGIN_SYMBOL;
}

wxString EffectAudioAlign::GetDescription() {
  return XO("Align selected tracks");
}

EffectType EffectAudioAlign::GetType() { return EffectTypeProcess; }

bool EffectAudioAlign::GetAutomationParameters(
    EffectAutomationParameters& parms) {
  return true;
}

bool EffectAudioAlign::SetAutomationParameters(
    EffectAutomationParameters& parms) {
  return true;
}

bool EffectAudioAlign::LoadFactoryDefaults() {
  return Effect::LoadFactoryDefaults();
}

bool EffectAudioAlign::Startup() { return true; }

bool EffectAudioAlign::Init() { return true; }

bool EffectAudioAlign::Process() {
  std::cout.precision(7);
  if (m_Bench) {
    RunBenchmarks();
  }

  SelectedTrackListOfKindIterator iter(Track::Wave, inputTracks());

  AudioAlignOptions options;

  const WaveTrack* reference_track = nullptr;
  for (Track* t = iter.First(); t; t = iter.Next()) {
    WaveTrack* wave_track = static_cast<WaveTrack*>(t);
    if (wave_track->GetChannel() == Track::RightChannel) {
      // Skip right channel. For now only process left channel or
      // mono tracks.
      continue;
    }
    std::cout << "Processing selected wave track" << wave_track->GetName()
              << std::endl;

    if (!reference_track) {
      reference_track = static_cast<WaveTrack*>(t);
      continue;
    }
    std::pair<double, double> alignment_result =
        AlignTracks(reference_track, wave_track, options);
    std::cout << "AlignTracks offset_s: " << alignment_result.first
              << " time_multiplier: " << alignment_result.second << std::endl;
    const double offset = alignment_result.first + reference_track->GetOffset();
    wave_track->SetOffset(offset);

    // TODO: If the detected speed is very close to 1 (actually, the
    // criterion should be the max time difference between the two
    // tracks), don't resample. It's likely to be a mistake from the
    // algorithm.
    if (alignment_result.second != 1) {
      ResampleTrack(1 / alignment_result.second, wave_track);
    }

    if (wave_track->GetLinked()) {
      // Also align the corresponding right channel.
      t = iter.Next();
      if (t) {
        wave_track->SetOffset(offset);
        ResampleTrack(1 / alignment_result.second, wave_track);
      }
    }
  }
  return true;
}

// Code mostly taken from ChangeSpeed.cpp. This should be refactored.
void EffectAudioAlign::ResampleTrack(double time_multiplier, WaveTrack* input) {
  std::cout << "Resampling track: " << input->GetName()
            << " multiplier: " << time_multiplier << std::endl;
  auto newTrack =
      mFactory->NewWaveTrack(input->GetSampleFormat(), input->GetRate());

  Resample resample(true, time_multiplier, time_multiplier);

  auto inBufferSize = input->GetMaxBlockSize();
  Floats inBuffer{inBufferSize};
  auto outBufferSize = size_t(time_multiplier * inBufferSize + 10);
  Floats outBuffer{outBufferSize};

  sampleCount start = input->TimeToLongSamples(input->GetStartTime());
  sampleCount end = input->TimeToLongSamples(input->GetEndTime());
  auto samplePos = start;
  while (samplePos < end) {
    auto blockSize = limitSampleBufferSize(input->GetBestBlockSize(samplePos),
                                           end - samplePos);
    input->Get((samplePtr)inBuffer.get(), floatSample, samplePos, blockSize);

    const auto results = resample.Process(
        time_multiplier, inBuffer.get(), blockSize,
        ((samplePos + blockSize) >= end), outBuffer.get(), outBufferSize);
    const auto outgen = results.second;

    if (outgen > 0) {
      newTrack->Append((samplePtr)outBuffer.get(), floatSample, outgen);
    }

    // Increment samplePos
    samplePos += results.first;
  }

  // Flush the output WaveTrack (since it's buffered, too)
  newTrack->Flush();

  // Take the output track and insert it in place of the original
  // sample data
  double newLength = newTrack->GetEndTime();
  LinearTimeWarper warper{input->GetStartTime(), input->GetStartTime(),
                          input->GetEndTime(),
                          input->GetStartTime() + newLength};
  input->ClearAndPaste(input->GetStartTime(), input->GetEndTime(),
                       newTrack.get(), true, false, &warper);
  std::cout << "Resampling done" << std::endl;
}

// This is test code that should go elsewhere.
void EffectAudioAlign::Benchmark(const wxString& dir_str,
                                 const AudioAlignOptions& options,
                                 std::vector<BenchmarkStats>* all_stats) {
  wxDir dir(dir_str);
  if (!dir.IsOpened()) {
    return;
  }
  std::cout << "Processing: " << dir_str << std::endl;
  std::vector<wxString> contents;
  wxString filename;
  bool cont = dir.GetFirst(&filename);
  while (cont) {
    contents.push_back(dir_str + "/" + filename);
    cont = dir.GetNext(&filename);
  }
  std::sort(contents.begin(), contents.end());
  bool leaf = true;
  for (const auto& c : contents) {
    std::cout << "CHILD: " << c.ToStdString() << std::endl;
    if (wxFileName::DirExists(c)) {
      Benchmark(c, options, all_stats);
      leaf = false;
    }
  }
  if (!leaf) {
    return;
  }
  if (contents.size() != 3) {
    std::cout << "Expected 3 items in dir: " << dir_str << " got "
              << contents.size() << std::endl;
    return;
  }

  /*
  if (dir_str.ToStdString().find("clock_drift_1.0001") == std::string::npos) {
    std::cout << "only processing drifts" << std::endl;
    return;
  }
  */

  double expected_shift = 0;
  std::unique_ptr<WaveTrack> track1, track2;
  BenchmarkStats stats;
  for (const auto& c : contents) {
    if (c.ToStdString().find("expected") != std::string::npos) {
      wxTextFile tfile;
      tfile.Open(c);
      wxString str = tfile.GetFirstLine();
      expected_shift = std::stod(str.ToStdString());
      continue;
    }
    TrackHolders newTracks;
    Tags tags;
    wxString errorMessage;
    bool success =
        Importer::Get().Import(c, mFactory, newTracks, &tags, errorMessage);
    if (!success || newTracks.size() != 2) {
      std::cout << "Error importing wave track " << c << std::endl;
      return;
    }
    if (!track1) {
      stats.a = c.ToStdString();
      track1 = std::move(newTracks[0]);
    } else {
      stats.b = c.ToStdString();
      track2 = std::move(newTracks[0]);
    }
  }
  wxASSERT(track1);
  wxASSERT(track2);
  double offset = AlignTracks(track1.get(), track2.get(), options).first;
  stats.error = fabs(offset - expected_shift);
  // 40ms. Note that inputs are manually-aligned, so there might be
  // small mistakes in the the test set.
  const double MAX_ERROR_SEC = 0.040;
  stats.success = stats.error < MAX_ERROR_SEC;
  std::cout << dir_str << " expected: " << expected_shift << " got: " << offset
            << " error: " << stats.error << " "
            << (stats.error < MAX_ERROR_SEC ? "SUCCESS" : "FAILURE")
            << std::endl;
  all_stats->push_back(stats);
}

struct SummarizedBenchStats {
  AudioAlignOptions bench_options;
  int num_failures;
  int num_success;
  // Counts failures by type of audio transforms.
  std::map<std::string, int> failures_by_transform;
  std::vector<BenchmarkStats> all_stats;
};

// This is test code that should go elsewhere.
void EffectAudioAlign::RunBenchmarks() {
  std::vector<SummarizedBenchStats> summarized_stats;
  for (auto summary_type : {
           SummaryType::kSumAbs, SummaryType::kMax,
           // SummaryType::kZeroCrossings,
           // SummaryType::kSpectralFlatness
       }) {
    for (auto shift_finding_method :
         {// ShiftFindingMethod::kMax,
          ShiftFindingMethod::kMaxPeakToAveragePowerRatio}) {
      //	for (auto correlation_normalization_power : {0., 0.2}) {
      for (auto correlation_normalization_power : {0.2}) {
        for (auto time_per_summarized_sample_millis : {25., 10.}) {
          // for (auto time_per_summarized_sample_millis : {25.}) {
          AudioAlignOptions bench_options;
          bench_options.summary_type = summary_type;
          bench_options.shift_finding_method = shift_finding_method;
          bench_options.correlation_normalization_power =
              correlation_normalization_power;
          bench_options.time_per_summarized_sample_millis =
              time_per_summarized_sample_millis;
          std::vector<BenchmarkStats> all_stats;
          Benchmark(wxString("/home/raphael/audio_alignment/benchmark_out"),
                    bench_options, &all_stats);
          int num_success = 0;
          std::map<std::string, int> failures_by_transform;
          for (const auto& s : all_stats) {
            num_success += s.success;
            failures_by_transform[GetTransformFromFilename(s.b)] += !s.success;
            std::cout << (s.success ? "SUCCESS: " : "FAILURE: ") << s.a << " "
                      << s.b << " error: " << s.error << std::endl;
            //	    std::cout << "TRANSFORM: " << GetTransformFromFilename(s.b)
            //<< std::endl;
          }
          std::cout << "NUM SUCCESSES: " << num_success
                    << " NUM_FAILURES: " << (all_stats.size() - num_success)
                    << std::endl;
          SummarizedBenchStats s;
          s.bench_options = bench_options;
          s.num_failures = all_stats.size() - num_success;
          s.num_success = num_success;
          s.all_stats = all_stats;
          s.failures_by_transform = failures_by_transform;
          summarized_stats.push_back(s);
          std::cout << "Bench: " << int(s.bench_options.summary_type) << " "
                    << int(s.bench_options.shift_finding_method) << " "
                    << s.bench_options.correlation_normalization_power << " "
                    << s.bench_options.time_per_summarized_sample_millis << " "
                    << " success: " << s.num_success
                    << " failures: " << s.num_failures
                    << " failure map: " << PrintMap(s.failures_by_transform)
                    << std::endl;
        }
      }
    }
  }

  for (auto s : summarized_stats) {
    std::cout << "Bench: " << int(s.bench_options.summary_type) << " "
              << int(s.bench_options.shift_finding_method) << " "
              << s.bench_options.correlation_normalization_power << " "
              << s.bench_options.time_per_summarized_sample_millis << " "
              << " success: " << s.num_success
              << " failures: " << s.num_failures
              << " failure map: " << PrintMap(s.failures_by_transform)
              << std::endl;
  }
}

void EffectAudioAlign::PopulateOrExchange(ShuttleGui& S) {
  S.SetBorder(5);

  S.StartVerticalLay(0);
  {
    S.AddSpace(0, 5);
    S.AddTitle(_("Align select tracks"));
  }
  S.EndVerticalLay();
}

bool EffectAudioAlign::TransferDataToWindow() { return true; }

bool EffectAudioAlign::TransferDataFromWindow() { return true; }
