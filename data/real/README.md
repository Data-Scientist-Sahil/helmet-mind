# Real test clips

Recorded by volunteers on ordinary phones, prompted by a **situation card** (never a script), e.g.:

> You're at the customer's building. The gate is locked. You've called twice. Leave a voice note for support.

- `labels.csv`: `clip_id, card_id, situation, action, safety, language, ambiguous, listener2_situation, release_ok`
- Each clip is labelled from its card, then by a second listener blind; disagreements are resolved or marked `ambiguous`.
- `release_ok` records the speaker's consent to publish. Only those clips are shared; `audio/` is git-ignored.
- These clips are used for testing only — never for training, tuning, thresholds or calibration.

## Recording kit

- `cards.csv` — 50 situation cards for the door-step wedge (18 safety, 5 deliberately ambiguous). Each volunteer
  gets a card, reads it, and leaves a voice note **in their own words**, in whatever mix of Hindi / English they
  naturally use. Never read the card out loud as the voice note.
- Record on a phone's voice-memo or messaging app. Mix conditions: about half outdoors near traffic, about 10 with a
  helmet on, about 10 while walking. 6–10 different speakers if possible.
- `labels_template.csv` — fill one row per clip: the card's labels, the language actually spoken, and a second
  listener's blind situation label. `release_ok` = the speaker agreed to publish the clip.
- Even 30–50 clips make a useful separate test tier; notebook 02 picks them up automatically when a dataset with
  `labels.csv` + `audio/` is attached.
