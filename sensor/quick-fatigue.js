// User-requested debugging proxy. These endpoints are arbitrary display-scale
// settings, not physiological fatigue thresholds or a personal resting baseline.
export class QuickFatigueScore {
  constructor() { this.last = null; this.latestAt = null; this.latestContactWarning = false; }

  observe(frame) {
    const t = frame?.recv_epoch_time_ms;
    if (!Number.isFinite(t) || (this.latestAt !== null && t <= this.latestAt)) return;
    this.latestAt = t;
    this.latestContactWarning = [frame.contact, frame.sensor_contact, frame.sensor_contact_detected].includes(false);
    if (Number.isFinite(frame.heart_rate_bpm) && frame.heart_rate_bpm > 0)
      this.last = { t, hr: frame.heart_rate_bpm, contactWarning: this.latestContactWarning };
  }

  snapshot(now) {
    const age = this.last ? now - this.last.t : null;
    const available = age !== null && age >= 0 && age <= 10000;
    const held = available && (this.latestAt > this.last.t || age > 2000);
    return {
      available, score: available ? Math.round(Math.max(0, Math.min(100, (this.last.hr - 60) / 80 * 100))) : null,
      source: 'hr-only-debug', provisional: true, usedSignals: ['reported-heart-rate'],
      hrBpm: available ? this.last.hr : null, measuredAtMs: this.last?.t ?? null,
      ageSeconds: age !== null ? Math.max(0, age / 1000) : null, held,
      contactWarning: Boolean(available && (this.last.contactWarning || this.latestContactWarning)),
      scale: { zeroAtBpm: 60, fullAtBpm: 140, maxAgeSeconds: 10 },
      reason: available ? 'HR-only test scale; not a validated fatigue measurement or probability.'
        : this.last ? 'No recent nonzero heart rate. Waiting for another HR packet.'
          : 'Waiting for the first nonzero heart-rate packet; no baseline is needed for quick test mode.',
    };
  }
}
