import { CalcCascades, IirFilter } from 'fili';

// Extracted from Yuhui Zhao's polar-h10-visualizer:
// src/consts.ts (initECGFilterSettings) and
// src/PolarH10VisualizerRow.ts (ecgFilterCoefReset / newECGCallback).
// https://github.com/yhzhao343/polar-h10-visualizer
// Source commit: 771559bc6ba6e2f13b43f02e1e7b026b1ec7473e (ISC).
const ECG_SAMPLE_RATE_HZ = 130;
const ECG_RMS_WINDOW_MS = 200;
const ECG_RMS_WINDOW_SIZE = Math.round(ECG_SAMPLE_RATE_HZ / (1000 / ECG_RMS_WINDOW_MS));
const initECGFilterSettings = {
  type: 'highpass', order: 4, characteristic: 'butterworth',
  Fs: ECG_SAMPLE_RATE_HZ, Fc: 25, Fl: 1, Fh: 20, BW: 19,
  gain: undefined, preGain: false,
};

export class PolarData {
  constructor() { this.reset(); }

  reset() {
    const coefficients = new CalcCascades().highpass(initECGFilterSettings);
    this.ecg_filter_iir = IirFilter(coefficients);
    this.ecg_mss_win = new Float64Array(ECG_RMS_WINDOW_SIZE);
    this.ecg_mss_win_i = this.ecg_buf_head = this.ecg_mss = 0;
    this.rms = [];
    this.lastReceived = null;
    this.lastTimestamp = null;
  }

  waveform(data) {
    if (!(data?.prev_sample_timestamp_ms > 0) || !data.samples ||
        !data.epoch_timestamps_ms || data.samples.length !== data.epoch_timestamps_ms.length) return;
    if (this.lastReceived !== null && !this.fresh()) this.reset();
    for (let s_i = 0; s_i < data.samples.length; s_i++) {
      const timestamp = data.epoch_timestamps_ms[s_i];
      const sample_i = data.samples[s_i];
      if (!Number.isFinite(timestamp) || !Number.isFinite(sample_i) ||
          (this.lastTimestamp !== null && timestamp <= this.lastTimestamp)) continue;
      this.lastTimestamp = timestamp;
      this.lastReceived = Date.now();
      // Upstream filtered-sample rolling RMS, with the same 200 ms window.
      const filtered_sample_i = this.ecg_filter_iir.singleStep(sample_i);
      const filtered_sample_squared_n_i = (filtered_sample_i / ECG_RMS_WINDOW_SIZE) * filtered_sample_i;
      this.ecg_mss += filtered_sample_squared_n_i;
      if (this.ecg_mss_win_i < ECG_RMS_WINDOW_SIZE) {
        this.ecg_mss_win[this.ecg_mss_win_i] = filtered_sample_squared_n_i;
        this.ecg_mss_win_i++;
      } else {
        this.ecg_mss -= this.ecg_mss_win[this.ecg_buf_head];
        this.ecg_mss_win[this.ecg_buf_head] = filtered_sample_squared_n_i;
        this.ecg_buf_head = (this.ecg_buf_head + 1) % ECG_RMS_WINDOW_SIZE;
      }
      if (this.ecg_mss_win_i === ECG_RMS_WINDOW_SIZE) {
        const rms = Math.sqrt(Math.max(0, this.ecg_mss));
        this.rms.push({ t: timestamp, values: [rms] });
      }
    }
    if (this.rms.length > 1950) this.rms.splice(0, this.rms.length - 1950);
  }

  fresh() {
    return this.lastReceived !== null && Date.now() - this.lastReceived <= 5000;
  }
}
