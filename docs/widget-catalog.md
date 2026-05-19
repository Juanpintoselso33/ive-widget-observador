# Widget Catalog — El Observador

Registro de widgets disponibles en la plataforma.

## Widgets activos

| Widget | Slug | Tema | Entry point | Estado |
|--------|------|------|-------------|--------|
| ¿Apoyás el IVE? | `ive` | IVE / aborto | `widgets/ive/app.py` | ✅ Activo |

## Widgets en desarrollo

_Ninguno aún._

## Crear un widget nuevo

Ver `widgets/_template/WIDGET_README.md`.

## Datasets disponibles

| Dataset | Ruta | Encuesta | N | Variables clave |
|---------|------|----------|---|-----------------|
| `base_limpia.csv` | `../base_limpia.csv` | El Observador 2025-2026 | ~3.300 | edad, sexo, educación, religiosidad, región, hijos, hogar, balotaje |

## Stack compartido

- `shared/styles.py` — CSS editorial (IBM Plex, Economist-style)
- `shared/config.py` — Paleta de colores + umbrales de interpretación
- `widgets/_template/` — Scaffold para widgets nuevos
