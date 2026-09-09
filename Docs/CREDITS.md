# Third-party asset credits

## VFX

**Explosion flipbook** (`/Game/Proto/VFX/T_ExplosionFlipbook`, and the
`NS_Explosion` / `NS_ExplosionTurtle` Fireball emitters built from it)

Baked from a third-party model. Attribution is **required** if this ships:

> This work is based on "Stylized Explosion Effect Simulation"
> (https://sketchfab.com/3d-models/stylized-explosion-effect-simulation-18cd925d7018432187bc854b158e418f)
> by SonicVisual (https://sketchfab.com/sonicvisual)
> licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

Commercial use is allowed. The source model is **not** imported into the project —
only a flipbook texture rendered offline from its vertex cache. Source files live in
`AssetsToImport/` and must never be imported directly (they OOM the editor).

The shockwave ring (`T_ShockRingFlipbook`) is generated procedurally by
`Docs/tools/vfx_bake/ring.py` and is original — no attribution needed.
