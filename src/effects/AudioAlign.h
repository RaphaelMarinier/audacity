/**********************************************************************

  Audacity: A Digital Audio Editor

  AudioAlign.h

  Raphael Marinier

  This is a prototype implementation of an Audacity tool that shifts
  (and changes the speed of) audio tracks so that their audio is
  aligned. The typical use-case is to align multiple recordings of the
  same event.

  The algorithm is as follows:

  1. Summarizes each track by computing a summary value for a sliding
     time window. Typically the window is 25ms. Summaries are much
     smaller than the original audio data, they can easily be stored
     in memory, and greatly speed up subsequent steps of the algorithm.
  2. Computing the cross-correlation between the summaries. This is
     done in O(n log n) with FFT thanks to the correlation theorem.
  3. Find the best shift from the cross-correlation function.
  4. If enabled, split summaries into small chunks, and align them
     1:1. This allows detecting small clock speed differences between
     devices. (If clock speed differences are too high, global
     alignment of steps 2/3 will probably fail). This step is O(n log
     k), where k is the chunk size (k << n typically).
  5. Apply the shift, and resample the track if need be.

**********************************************************************/

#ifndef __AUDACITY_EFFECT_AUDIOALIGN__
#define __AUDACITY_EFFECT_AUDIOALIGN__

#include <wx/choice.h>
#include <wx/event.h>
#include <wx/slider.h>
#include <wx/string.h>
#include <wx/textctrl.h>

#include "../widgets/NumericTextCtrl.h"

#include "Effect.h"

class ShuttleGui;

#define AUDIOALIGN_PLUGIN_SYMBOL XO("Align selected tracks")
#define AUDIOALIGN_BENCHMARK_PLUGIN_SYMBOL XO("Align selected tracks benchmark")

// Controls the algorithm used to summarized each window of audio
// sample data.
enum class SummaryType {
  // The summary is sum(fabs(s_i)) where s_i are the samples in the
  // window considered.
  kSumAbs,
  // The summary is max(fabs(s_i)).
  kMax,
  // For each window, count the number of times s_{i+1} and s_{i} have
  // different signs. See
  // https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42193.pdf
  // for more details.
  kZeroCrossings,
  // Spectral flatness feature (also called Wiener entropy). For
  // each window, the power spectrum is computed. Then the summary is:
  // geometric_mean(spectrum_f) / arithmetic_mean(spectrum_f)
  // where spectrum_f are the frequencies of the power spectrum.
  // See https://en.wikipedia.org/wiki/Spectral_flatness.
  // This will be 1 for white noise, and 0 for a pure tone.
  kSpectralFlatness,
};

// Method to find the best shift from the cross-correlation function.
enum class ShiftFindingMethod {
  // Shift is argmax(cross_i) where cross_i are value of the
  // cross-correlation function.
  kMax,
  // This method tries to find the position of a narrow peak in the
  // cross-correlation function, even though it's not a global
  // maximum. See
  // https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42193.pdf.
  // In practice this seems to work better than just finding the global maximum.
  kMaxPeakToAveragePowerRatio
};

struct AudioAlignOptions {
  SummaryType summary_type = SummaryType::kSumAbs;
  ShiftFindingMethod shift_finding_method = ShiftFindingMethod::kMaxPeakToAveragePowerRatio;
  // This only applies when using the shift finding method
  // kMaxPeakToAveragePowerRatio. The standard
  // max-peak-to-average-power ratio method finds 'i' that maximizes:
  // cross_i ^ 2 / sum(cross_j^2, |i-j| < K), for some parameter K.
  // Sometimes the peak found has very low amplitude. In order to
  // favor higher amplitude peaks, we add an additional term and maximize:
  // cross_i ^ 2 / sum(cross_j^2, |i-j| < K) *
  //     cross_i ^ correlation_normalization_power
  // instead.
  double correlation_normalization_power = 0.2;
  // This is the duration of each summarized time window. Expressed in
  // time and not number of samples allows windows to be comparable
  // across multiple tracks with different sample rates.
  double time_per_summarized_sample_millis = 25;
  // If true, we try to identify and correct (small) clock speed
  // differences between track, which can happen when the tracks were
  // recorded by different devices.
  bool correct_time_drift = true;
  // Size of the windows in seconds aligned in order to identify the
  // time drift. Only relevant when 'correct_time_drift' is true.
  double time_drift_detection_window_s = 60;
};

struct BenchmarkStats {
  std::string a, b;
  double error = 0;
  bool success = false;
};

class EffectAudioAlign final : public Effect
{
 public:
   // If 'bench' is true, a benchmark will run when Process() is
   // called. This is for testing.
   EffectAudioAlign(bool bench);
   virtual ~EffectAudioAlign();

   wxString GetSymbol() override;
   wxString GetDescription() override;

   EffectType GetType() override;

   // EffectClientInterface implementation

   bool GetAutomationParameters(EffectAutomationParameters & parms) override;
   bool SetAutomationParameters(EffectAutomationParameters & parms) override;
   bool LoadFactoryDefaults() override;

   // Effect implementation

   bool Startup() override;
   bool Init() override;
   bool Process() override;
   void PopulateOrExchange(ShuttleGui & S) override;
   bool TransferDataFromWindow() override;
   bool TransferDataToWindow() override;

 private:
   // Resamples the given track in place, by the given factor.
   void ResampleTrack(double time_multiplier,
		      WaveTrack* input);

  // Benchmark: performs audio alignment of all files in the directory
  // (incl. subdirs, recursively), compares with the "expected" file,
  // and reports stats.
  void Benchmark(const wxString& dir,
		const AudioAlignOptions& options,
		std::vector<BenchmarkStats>* all_stats);

  // Runs the benchmark for multiple combinations of options, and
  // aggregate performance stats.
  void RunBenchmarks();

   // True when the aligner is in benchmark mode.
   bool m_Bench;

   DECLARE_EVENT_TABLE()
};

#endif // __AUDACITY_EFFECT_AUDIOALIGN__
