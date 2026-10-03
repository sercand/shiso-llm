# Shiso Model Registry

This repository is the source of the Shiso model registry. It is served
statically as-is: a client configured with `https://<host>/` finds the index
at `/registry/v1/index.json` and the compiled manifests under
`/registry/v1/models/`.

This directory is currently staged inside the shiso worktree as
`model-registry/`; its contents are the root of the standalone registry repo.

## Structure

```text
.
├── README.md               # this document
├── index.html              # landing page for the static host
├── error.html              # error page for the static host
├── recipes/<name>.json     # authored recipes (no timestamps)
└── registry/v1/            # built artifacts, committed; served as-is
    ├── index.json          # index; each entry pins its manifest's sha256
    └── models/<name>.json  # compiled manifests with Git-derived timestamps
```

## Adding a Recipe

1. Create a new manifest file in `recipes/<name>.json`.
2. Ensure `name` matches lowercase `[a-z0-9][a-z0-9._-]*`.
3. Fill in supported `variants`, their backend (`metal` or `cuda`),
   quantization labels, and components.
4. Each component lists remote files from Hugging Face with commit-pinned
   40-character SHAs and 64-character SHA-256 digests.
5. Validate your recipe (run from this directory, or pass `--repo`):
   ```sh
   cargo run -p shiso-registry-editor -- validate --repo .
   ```

## Building the Registry

To compile the manifests into `registry/v1/` and regenerate the index:

```sh
cargo run -p shiso-registry-editor -- build --repo .
```

The build derives `published_at` and `updated_at` timestamps from Git history
and refuses to run with uncommitted changes under `recipes/`. Commit the
rebuilt `registry/v1/` alongside the recipe change.

## Publishing

Publishing is `git push`: the static host serves this repository, so a pushed
commit to the default branch is the release. Integrity is digest-pinned, not
signed: `index.json` records the SHA-256 of every compiled manifest, and every
model file carries its own `sha256`, which the client verifies after download.
Transport security is HTTPS; the registry host is the trust anchor.
