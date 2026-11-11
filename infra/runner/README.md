# Self-hosted CUDA runner — bring-up

`cloud-init.yaml` here provisions a GitHub Actions self-hosted runner with
a real NVIDIA GPU (T4 / `sm_75` by default, A10G / `sm_86` if you swap
the instance type).  It is the runner side of the workflow defined in
`.github/workflows/cuda.yml`.

## Pick an instance

| Instance         | GPU       | Arch    | Spot $/hr (us-east-1) | Notes                        |
| ---------------- | --------- | ------- | --------------------- | ---------------------------- |
| `g4dn.xlarge`    | T4 16 GB  | sm_75   | ~$0.16                | Cheapest practical default   |
| `g5.xlarge`      | A10G 24GB | sm_86   | ~$0.50                | ~2× faster builds            |
| Own hardware     | any       | any     | $0                    | Skip the AWS bits, keep the runner config |

`cuda.yml` pins `-DCMAKE_CUDA_ARCHITECTURES=75` so other contributors with
different cards do not get phantom skips when CI rebuilds.  If you switch
to `g5.xlarge`, also bump the workflow's `CMAKE_CUDA_ARCHITECTURES` to
`86`.

## One-time launch

1. In the GitHub UI: repo Settings → Actions → Runners → **New self-hosted
   runner** → choose Linux/x64 → copy the token (valid 1h).
2. Launch an EC2 spot instance from the latest Ubuntu 22.04 LTS AMI:
   - Instance type: `g4dn.xlarge`
   - Storage: 50 GB gp3 (CUDA toolkit + CPM cache fit comfortably)
   - IAM role: needs `ec2:TerminateInstances` on `self` (so the budget
     guard in cloud-init.yaml can self-terminate after 2h idle)
   - **Launch template `MetadataOptions`**: set
     `HttpTokens: required, HttpPutResponseHopLimit: 2`.
     This enforces IMDSv2 on the instance and is required because
     `cloud-init.yaml` uses IMDSv2 token-based metadata fetches.
     A self-hosted runner that executes PR code is an SSRF target; IMDSv2
     prevents PR code from stealing IAM credentials via the metadata API.
   - User data: paste `cloud-init.yaml`, then in the same field, set the
     three required env vars at the very top **before** the `#cloud-config`
     line, e.g.:
     ```bash
     #!/bin/bash
     export RUNNER_REPO_URL=https://github.com/NamecoinGithub/NexusMiner
     export RUNNER_REGISTRATION_TOKEN=<paste from step 1>
     export RUNNER_NAME=nexusminer-cuda-1
     # ... then the rest of cloud-init.yaml as a heredoc, OR use a
     # multi-part MIME user-data so cloud-init parses both the shell
     # block and the cloud-config block.
     ```

   For a less fragile setup, bake an AMI from the cloud-init result and
   launch from that with the env vars supplied via systemd drop-in.

3. The runner registers itself with labels
   `self-hosted, linux, x64, gpu, cuda` and immediately starts polling
   for jobs.  `cuda.yml` targets that exact label set.

## Why these specific guardrails

The `runner-budget-guard` cron exists because **the most expensive failure mode
for a self-hosted runner is a stuck or orphaned one** — not a hot build.  A
wedged build that takes 45 minutes costs ~$0.12; a runner that drains 24h of
idle spot time before someone notices costs ~$4.  The budget guard caps the
worst case at ~$0.32 (2h spot + termination).

Note: the `actions-runner-shutdown.service` that was described in earlier
versions of this file has been **removed** because it relied on a placeholder
removal token (`config.sh remove --token placeholder`) that always failed,
leaving "offline" ghost runners in the GitHub UI for ~24h after every spot
interruption.  A proper fix requires a PAT or JIT remove-token stored securely
(e.g. AWS Secrets Manager / SSM Parameter Store) — see the PR #682 review for
the design sketch.  Until that wiring is in place, orphaned runners expire
automatically after 24h, which is acceptable given the budget guard already
covers the cost risk.

## Trust boundary

`cuda.yml` only runs on PRs from same-repo branches by default (the
`if:` on `cuda-build`).  Fork PRs require a maintainer to manually run
the workflow via `workflow_dispatch` after reviewing the diff.  Do **not**
loosen this — a self-hosted runner running unreviewed fork code is the
canonical way to get the host owned.

## Promoting to a required check

`cuda.yml` currently sets `continue-on-error: true` so that flakes from
driver/spot-interruption noise during the warm-up period do not block
CPU-side work.  After ~2 weeks of clean runs, flip that to `false` and
add the `cuda-build` check to the branch protection rule.
