# Mining setup

Start with **one** of these solo-mining templates. Both connect to a Nexus node
on the same computer; neither is a pool configuration.

| What you want to mine | Required binary | Start with |
|---|---|---|
| Channel 1: CPU Prime | CPU Prime build for your OS | `cpu-prime-solo.config` |
| Channel 2: NVIDIA GPU Hash | NVIDIA CUDA hash build for Linux or Windows | `gpu-hash-solo.config` |

A CPU-only binary cannot use an NVIDIA GPU. The NVIDIA **hash-only** release
does not support Prime, even on the CPU. AMD GPU Hash and macOS GPU Hash are
not implemented. GPU mining must be tested on your own hardware; a successful
build does not establish that it mines correctly.

## Set up and run

1. Start a synchronized Nexus node with mining enabled and the mining wallet
   unlocked. These examples use `127.0.0.1:8323`, the default legacy mining
   lane, for **both channels**. If your node uses a different mining port,
   change `[wallet] port` to match it. `9323` is the stateless lane when your
   node supports it; ports are protocol lanes, not channel numbers.
2. In your wallet, run `system/get/info` and copy the 64-character `genesis`
   hash. Use that value for both `genesis` and `reward_address` to receive
   rewards on the same Tritium sigchain. The reward address here is a genesis
   hash, **not** a Base58 account address.
3. Generate a Falcon key pair with `./NexusMiner --create-keys` on Linux/macOS,
   or `.\NexusMiner.exe --create-keys` in Windows PowerShell. Copy the full
   PUBLIC KEY and PRIVATE KEY hex strings into `[falcon] pubkey` and `privkey`
   respectively. Ignore the JSON snippet printed by the command when editing
   these `.config` files. If the node uses a miner key whitelist, its operator
   must authorize the public key. **Never share the private key**.
4. Copy the chosen template to `miner.config` next to the executable and edit
   the copy. For CPU Prime, adjust `[workers] count` for your CPU. For GPU
   Hash, start with `count = 1` and `[gpu] device = 0`. Keep the `.config`
   extension: `.conf` is parsed as JSON, not TOML.
5. Protect `miner.config`: on Linux/macOS run `chmod 600 miner.config`; on
   Windows restrict its file permissions to your account. Do not upload or
   share it after adding your private key.
6. From the directory containing the executable and your config, run:

   ```sh
   ./NexusMiner miner.config
   ```

   In Windows PowerShell use `.\NexusMiner.exe .\miner.config`. Stop with
   Ctrl+C. Check the log for the selected channel and worker, node connection,
   authentication, and work arriving before relying on mining statistics.

Do not use `-c` to check a `.config` file: the current check accepts JSON
only, and a successful check continues into mining rather than exiting.
If you see an authentication failure, check the full key pair, genesis and
any node whitelist; do not post your private key when asking for help.

## Other examples

Most miners only need the two files above. The other templates are retained
for reference, not as additional required steps:

- `cpu-hash-solo.config`: CPU Hash, useful for testing.
- `cpu-prime-performance.config` and `gpu-hash-performance.config`: alternate
  starting values, not guaranteed performance improvements.
- `gpu-prime-solo.config`: experimental; requires a GPU Prime-enabled binary.
- `cpu-gpu-*-hybrid.config`: compatibility examples; despite their filenames,
  each creates only one worker type. See comments in each file.
- `ssl-remote-*.config`: historical reference only. The current mining
  connection does not use these TLS settings; they do **not** make a remote
  connection encrypted. Use a secure tunnel for remote mining.
- `MASTER-REFERENCE.config`: options reference, not a runnable profile.

For multiple NVIDIA GPUs in a `.config` file, set `[workers] count` to the
number of GPUs and `[gpu] device` to the first device index: indices are
assigned consecutively (for example 0, 1, 2). A `.config` file selects one
worker hardware type; mixed CPU/GPU workers require the JSON configuration
format or separately configured instances.

See [BUILD.md](../BUILD.md) for building binaries and the
[configuration reference](https://github.com/NamecoinGithub/NexusMiner/blob/MINER/docs/reference/nexus.conf.md)
for advanced options (the reference is online, not included in the archives).
