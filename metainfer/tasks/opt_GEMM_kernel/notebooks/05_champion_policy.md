# Champion/challenger policy

Every iteration starts from the persisted champion, not merely the most recent
candidate. A challenger is promoted only after compile, complete correctness,
multi-shape scoring and critical-regression gates pass.

The challenger must also exceed the champion's weighted speedup by the noise
threshold. Failed and non-promoted candidates remain in iteration history for
diagnosis, but they never become the starting implementation for the next
iteration.

At the end of the task, `state/champion/submission/` is the selected artifact
and `champion.json` identifies its source iteration and score.

