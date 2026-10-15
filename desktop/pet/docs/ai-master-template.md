# AI Live2D master template

tulpa does not auto-rig a new Live2D model. It preserves an owned Cubism
rig and replaces only fixed texture-atlas regions. This keeps the `.moc3`,
motions, expressions, physics, hit areas, and lip-sync parameters deterministic.

An installable mother-template folder must contain:

- exactly one complete Cubism model referenced by `modelEntry`;
- fixed Moc, texture, physics, motion, and expression references;
- a valid schema-v2 `pet.manifest.json`;
- `master.template.json`;
- every RGBA silhouette mask referenced by the template manifest.

Example manifest:

```json
{
  "schemaVersion": 1,
  "id": "owned-fullbody-girl-v1",
  "name": "Owned Full-body Girl V1",
  "modelEntry": "girl/girl.model3.json",
  "textureAtlases": [
    { "file": "girl/textures/texture_00.png", "width": 2048, "height": 2048 }
  ],
  "slots": [
    {
      "id": "body",
      "name": "完整身体与服装",
      "required": true,
      "zIndex": 10,
      "atlas": { "texture": 0, "x": 0, "y": 0, "width": 1024, "height": 2048 },
      "anchor": { "x": 512, "y": 1024 },
      "silhouetteMask": "masks/body.png",
      "completionPrompt": "complete the full body and clothing without occlusion",
      "children": [
        {
          "id": "face",
          "name": "脸部皮肤",
          "required": true,
          "zIndex": 20,
          "atlas": { "texture": 0, "x": 1024, "y": 0, "width": 512, "height": 512 },
          "anchor": { "x": 1280, "y": 256 },
          "silhouetteMask": "masks/face.png",
          "completionPrompt": "complete the unobstructed face skin, ears, and jaw",
          "children": []
        }
      ]
    }
  ]
}
```

`anchor` uses absolute atlas coordinates. A slot rectangle must remain inside
its declared texture. Slot IDs are globally unique. Paths are safe relative
paths and cannot contain `..`. The recursive generation planner stops at depth
three and twelve decomposition calls. It uses a uniform scale plus translation
around the declared anchor; an incompatible aspect ratio is rejected instead
of being stretched into the rig. Each silhouette mask must be a valid RGBA PNG
whose dimensions exactly match its slot rectangle; its alpha channel clips the
generated pixels during final atlas composition.

The confirmation preview automatically demonstrates left/right gaze, AEIOU
mouth shapes, the first available authored motion, and then returns to standby.
Cubism's authored blink, breathing, physics, and pose continue to run during
the sequence.

The example describes the format only. Do not add licensed official models,
Epsilon, user characters, fal downloads, or paid acceptance fixtures to Git.

## Local Epsilon development template

After accepting Live2D's sample-data terms and downloading Epsilon, a local
development template can be prepared from the Free runtime folder without
copying any Live2D assets into Git:

```sh
npm run prepare:epsilon-master -- /path/to/epsilon_free/runtime /path/to/output
```

The helper copies the runtime model, creates tulpa's manifests, and
derives exact RGBA slot masks from the official 2048 texture atlas. The output
is for local pipeline validation only and must not be bundled with the app.
