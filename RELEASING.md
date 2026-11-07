# Releasing a new version

Each GitHub release is archived automatically on Zenodo (all versions: [10.5281/zenodo.23945569](https://doi.org/10.5281/zenodo.23945569)).

1. Update the version and `date-released` in `.zenodo.json` and `CITATION.cff`, and the licence notes in `README.md` and `DATA_LICENSES.md` if they change.
2. Update `MANIFEST.json` (path, size and SHA-256 of every tracked file except itself):

   ```bash
   python3 - <<'PY'
   import hashlib, json, subprocess
   paths = sorted(p for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines() if p != "MANIFEST.json")
   entries = [{"path": p, "bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}
              for p in paths for d in [open(p, "rb").read()]]
   with open("MANIFEST.json", "w", newline="\n") as f:
       f.write(json.dumps(entries, indent=1) + "\n")
   PY
   ```

3. Commit, then create a GitHub release with tag `vX.Y.Z` and release notes that state the changes since the previous version and the licences.
4. On Zenodo, open the new version created from the release. Zenodo takes a single licence from `.zenodo.json` (MIT), so add the second one: **Edit → Licenses → Add standard → Creative Commons Attribution Share Alike 4.0 International → Publish**. Check that the published record lists both licences.
5. If files mirrored on Hugging Face ([securagenai/signed-but-steerable](https://huggingface.co/datasets/securagenai/signed-but-steerable)) change, upload them and update the dataset card. The answers generated with Qwen2.5-3B-Instruct (`results/e2/gen_*.jsonl`) and `data/prior/` are not mirrored there.
