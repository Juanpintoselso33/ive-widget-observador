# Widget Catalog — El Observador

Registro de widgets disponibles en la plataforma.

## Widgets activos

| Widget | Slug | Tema | Entry point | Estado |
|--------|------|------|-------------|--------|
| ¿Apoyás el IVE? | `ive` | IVE / aborto | `widgets/ive/app.py` | ✅ Activo |

Desplegado en **https://juanpintoselso33-ive-widget-observador-app-poekzu.streamlit.app/**,
con *Main file path* `app.py` — que desde el 19/9/2026 no repite el widget: ejecuta
`widgets/ive/app.py`, que es el entry real.

**Usa la hoja del Figma** (`get_observador_css`), la misma que seguridad, desde el
19/9/2026. Es el widget que la diseñadora dibujó: el frame de `Producto UY` tiene
sus campos. Ver `docs/diseno/figma-producto-uy.md`.

## Widgets en desarrollo

| Widget | Slug | Tema | Entry point | Estado |
|--------|------|------|-------------|--------|
| Mano dura, perfil por perfil | `seguridad` | Seguridad / punitivismo | `widgets/seguridad/app.py` | ✅ Activo |

El de seguridad publica **cuatro preguntas a la vez** y el lector elige cuál
estimar con un selector: votar a un político de mano dura, cadena perpetua por
tres delitos, pena de muerte por homicidio y humillación a los presos. Hay un
modelo entrenado por pregunta, en `widgets/seguridad/modelos/`. Ver
`widgets/seguridad/WIDGET_README.md`.

Desplegado en **https://observador-seguridad.streamlit.app/** — es una app aparte de la del IVE, sobre el MISMO repo,
con *Main file path* `widgets/seguridad/app.py`. Sigue la rama `master`, así que
publica lo que esté mergeado. El `app.py` de la raíz sirve el widget IVE.

Código de embed (nota, portada y variante con altura automática, con las alturas
medidas): `docs/embed/seguridad-widget-embed.html`.

## Crear un widget nuevo

Ver `widgets/_template/WIDGET_README.md`.

## Datasets disponibles

| Dataset | Ruta | Encuesta | N | Variables clave |
|---------|------|----------|---|-----------------|
| `base_limpia.csv` | `../base_limpia.csv` | El Observador 2025-2026 | ~3.300 | edad, sexo, educación, religiosidad, región, hijos, hogar, balotaje |
| `base_etiquetada.csv` | repo `Observador-encuesta`, `encuestas/observador_2026_05_seguridad/output/` | El Observador — Seguridad, mayo 2026 | 3.377 | edad, sexo, educación, región, ideología 0-10, víctima de delito, batería punitiva (var_208 a var_242) |

Ninguno de los dos vive en este repo: son datos del cliente y el `.gitignore`
excluye `*.csv`. El widget de seguridad acepta `SEGURIDAD_DATA_FILE` para
apuntar a otra ruta.

## Stack compartido

- `shared/styles.py` — CSS editorial (IBM Plex, Economist-style)
- `shared/config.py` — Paleta de colores + umbrales de interpretación
- `widgets/_template/` — Scaffold para widgets nuevos
