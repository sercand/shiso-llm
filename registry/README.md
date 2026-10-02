# Shiso Model Registry

This directory contains the source recipes and public verification key for the Shiso model registry (`https://models.shiso.run/v1/`).

## Structure

```text
registry/
├── README.md               # this document
├── recipes/<name>.json     # authored recipes (no timestamps)
└── keys/registry.pub       # Ed25519 public key (matches constant in shiso-hub)
```

## Adding a Recipe

1. Create a new manifest file in `registry/recipes/<name>.json`.
2. Ensure `name` matches lowercase `[a-z0-9][a-z0-9._-]*`.
3. Fill in supported `variants`, their backend (`metal` or `cuda`), quantization labels, and components.
4. Each component lists remote files from Hugging Face with commit-pinned 40-character SHAs and 64-character SHA-256 digests.
5. Validate your recipe:
   ```sh
   cargo run -p shiso-registry-editor -- validate
   ```

## Building and Signing the Registry

To compile the manifests into `target/registry/v1/` and generate the signed `index.json`:

```sh
cargo run -p shiso-registry-editor -- build --sign-key ~/.config/shiso/registry.key
```

The build derives `published_at` and `updated_at` timestamps from Git history and refuses to run with uncommitted changes under `registry/recipes/`.

## Publishing

Upload compiled manifests and signed index to the static host:

```sh
bash scripts/registry_publish.sh target/registry/v1/
```
