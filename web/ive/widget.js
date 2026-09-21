/* ===========================================================================
   Widget IVE — El Observador. Versión estática.

   Todo corre en el navegador: la predicción es una regresión logística de 19
   predictores, o sea una suma y una exponencial. No hace falta servidor, y por
   eso esta versión no se duerme ni se cae con el tráfico de una nota.

   LA INFERENCIA ES UN PORTE LITERAL de `widgets/ive/model.py`. Hay un test que
   recorre las 5.760 combinaciones posibles de perfil y exige que Python y JS
   den el mismo número: `web/ive/tests/paridad.py`. Si se toca un coeficiente o
   una dummy, ese test es el que avisa.
   =========================================================================== */

(function () {
  "use strict";

  // La inferencia vive en modelo.js: el test de paridad la carga en Node y la
  // compara contra model.py. Acá sólo se usa.
  var M = window.ModeloIVE;

  // Las dimensiones del bloque comparativo y la etiqueta de cada grupo.
  var GRUPOS_ORDEN = [
    ["Por religiosidad", ["religiosidad_nada", "religiosidad_poco", "religiosidad_bastante", "religiosidad_mucho"]],
    ["Por balotaje 2024", ["balotaje_martinez", "balotaje_lacalle"]],
    ["Por educación", ["educacion_primaria", "educacion_secundaria", "educacion_ter_incomp", "educacion_ter_comp"]],
    ["Por edad", ["edad_18-24", "edad_25-34", "edad_35-44", "edad_45-54", "edad_55+"]]
  ];

  var GRUPOS_LABEL = {
    "religiosidad_nada": "Nada religioso",
    "religiosidad_poco": "Poco religioso",
    "religiosidad_bastante": "Bastante religioso",
    "religiosidad_mucho": "Muy religioso",
    "balotaje_martinez": "Orsi (FA)",
    "balotaje_lacalle": "Delgado (Coalición)",
    "educacion_primaria": "Primaria o menos",
    "educacion_secundaria": "Secundaria",
    "educacion_ter_incomp": "Terciaria incompleta",
    "educacion_ter_comp": "Terciaria completa+",
    "edad_18-24": "18-24",
    "edad_25-34": "25-34",
    "edad_35-44": "35-44",
    "edad_45-54": "45-54",
    "edad_55+": "55+"
  };

  // Claves que el modelo trae y la grilla NO publica, con el motivo. Igual que
  // en components.py: sin esta lista, una dimensión nueva quedaría invisible y
  // una dejada afuera a propósito no se distinguiría de un olvido.
  var GRUPOS_NO_PUBLICADOS = ["hogar_1_2", "hogar_3_4", "hogar_5_plus"];

  // ---------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------

  function crear(tag, clase, texto) {
    var el = document.createElement(tag);
    if (clase) el.className = clase;
    if (texto !== undefined) el.textContent = texto;
    return el;
  }

  function construirCampos(contenedor, rangos, alCambiar) {
    // [clave del estado, etiqueta, clave en variable_ranges, ayuda]
    // Las ayudas son las mismas que los `help=` de los selectbox de la versión
    // Streamlit: se perdían al portar y son las que explican qué se pregunta.
    var CAMPOS = [
      ["tramoEdad", "Edad", "tramo_edad_num", "Selecciona tu tramo de edad"],
      ["religiosidad", "Religiosidad", "religiosidad_num", "¿Cuán religioso/a te consideras?"],
      ["esMujer", "Sexo", "es_mujer", "Selecciona tu sexo"],
      ["esMontevideo", "Región", "es_montevideo", "¿Dónde vives?"],
      ["nivelEduc", "Nivel educativo", "nivel_educ_num", "Selecciona tu nivel educativo más alto"],
      ["tieneHijos", "¿Tienes hijos?", "tiene_hijos", "¿Tienes hijos/as?"],
      ["balotaje", "Balotaje 2024", "balotaje", "¿A quién votaste en el balotaje de 2024?"],
      ["hogar", "Personas en el hogar", "hogar_num", "Cantidad de personas que viven en tu hogar"]
    ];

    // El orden de arriba es el de LECTURA por filas, que es como los ve el
    // lector en la grilla de dos columnas: Edad y Religiosidad en la primera
    // fila, Sexo y Región en la segunda, etc. Es el mismo reparto que la
    // versión Streamlit, donde una columna tenía edad/sexo/educación/balotaje
    // y la otra religiosidad/región/hijos/hogar.
    CAMPOS.forEach(function (campo) {
      var clave = campo[0], etiqueta = campo[1], rango = rangos[campo[2]];
      var div = crear("div", "campo");
      var id = "campo-" + clave;

      var lab = crear("label", null, etiqueta);
      lab.setAttribute("for", id);
      if (campo[3]) {
        // El signo de pregunta con la ayuda. `title` da el tooltip nativo; el
        // `aria-label` es para que un lector de pantalla lo anuncie, porque el
        // carácter solo no dice nada.
        var ayuda = crear("span", "ayuda", "?");
        ayuda.title = campo[3];
        ayuda.setAttribute("aria-label", campo[3]);
        ayuda.setAttribute("role", "img");
        lab.appendChild(ayuda);
      }

      var sel = document.createElement("select");
      sel.id = id;
      sel.dataset.clave = clave;

      rango.labels.forEach(function (texto, i) {
        var opt = document.createElement("option");
        opt.textContent = texto;
        opt.value = String(i);
        sel.appendChild(opt);
      });

      sel.value = String(M.indicePorDefecto(rango));

      sel.addEventListener("change", alCambiar);
      div.appendChild(lab);
      div.appendChild(sel);
      contenedor.appendChild(div);
    });
  }

  function leerPerfil(contenedor, rangos) {
    function idx(clave) {
      return parseInt(contenedor.querySelector('[data-clave="' + clave + '"]').value, 10);
    }
    // La conversión en sí vive en modelo.js, pura, para que el test la ejerza.
    return M.perfilDesdeIndices(rangos, {
      tramoEdad: idx("tramoEdad"), esMujer: idx("esMujer"),
      nivelEduc: idx("nivelEduc"), religiosidad: idx("religiosidad"),
      esMontevideo: idx("esMontevideo"), tieneHijos: idx("tieneHijos"),
      hogar: idx("hogar"), balotaje: idx("balotaje")
    });
  }

  function pintarResultado(nodos, prob) {
    var probR = M.redondear(prob);
    nodos.indicador.style.left = prob + "%";
    nodos.pastilla.textContent = probR + "%";
    nodos.numero.textContent = probR + "%";
    nodos.texto.innerHTML =
      "<strong>Es " + M.textoInterpretacion(prob) + "</strong> al IVE según tus " +
      "características, pero esto es un ejercicio de probabilidades y no una " +
      "confirmación de tus posiciones.";
  }

  function construirComparacion(raiz, modelo) {
    var stats = modelo.stats_by_group;
    var nacionalR = M.redondear(modelo.prob_nacional);

    var dimensiones = GRUPOS_ORDEN.map(function (dim) {
      var celdas = dim[1].filter(function (k) {
        return stats[k] !== undefined && stats[k] !== null;
      }).map(function (k) { return [k, stats[k]]; });
      return [dim[0], celdas];
    }).filter(function (dim) { return dim[1].length > 0; });

    if (!dimensiones.length) return;

    var solapas = crear("div", "solapas");
    solapas.setAttribute("role", "tablist");
    solapas.setAttribute("aria-label", "Dimensiones de comparación");
    var paneles = [];
    var botones = [];

    function activar(indice) {
      botones.forEach(function (b, j) {
        b.setAttribute("aria-selected", j === indice ? "true" : "false");
        b.tabIndex = j === indice ? 0 : -1;
      });
      paneles.forEach(function (p, j) { p.hidden = j !== indice; });
    }

    dimensiones.forEach(function (dim, i) {
      var boton = crear("button", "solapa", dim[0]);
      boton.type = "button";
      boton.setAttribute("role", "tab");
      boton.setAttribute("aria-selected", i === 0 ? "true" : "false");
      // Un solo tab en el orden de foco, y las flechas mueven entre ellas: es
      // lo que espera quien navega por teclado cuando encuentra `role="tab"`.
      // Estaban las tres cosas a medias —todas enfocables, sin flechas y sin
      // relación con su panel—, que es peor que no declarar el rol. Lo marcó
      // Codex.
      boton.tabIndex = i === 0 ? 0 : -1;
      boton.id = "solapa-" + i;
      boton.setAttribute("aria-controls", "panel-" + i);

      var panel = crear("div", "grupo-cifras");
      panel.setAttribute("role", "tabpanel");
      panel.id = "panel-" + i;
      panel.setAttribute("aria-labelledby", "solapa-" + i);
      if (i !== 0) panel.hidden = true;

      dim[1].forEach(function (par) {
        var clave = par[0], valor = par[1];
        var celda = crear("div", "grupo-celda");
        celda.appendChild(crear("div", "grupo-celda-label", GRUPOS_LABEL[clave]));
        celda.appendChild(crear("div", "grupo-celda-valor", M.redondear(valor) + "%"));

        // Sobre los valores YA redondeados, que son los que se ven: restar
        // antes y redondear después deja cuentas que no cierran a la vista.
        var d = M.redondear(valor) - nacionalR;
        var delta;
        if (d !== 0) {
          var signo = d > 0 ? "+" : "−";
          delta = crear("div", "grupo-celda-delta grupo-celda-delta--" + (d > 0 ? "sube" : "baja"),
                        signo + Math.abs(d) + "pp");
        } else {
          delta = crear("div", "grupo-celda-delta", "igual al promedio");
        }
        celda.appendChild(delta);
        panel.appendChild(celda);
      });

      boton.addEventListener("click", function () { activar(i); });
      boton.addEventListener("keydown", function (e) {
        var salto = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
        var destino = null;
        if (salto) destino = (i + salto + dimensiones.length) % dimensiones.length;
        else if (e.key === "Home") destino = 0;
        else if (e.key === "End") destino = dimensiones.length - 1;
        if (destino === null) return;
        e.preventDefault();
        activar(destino);
        botones[destino].focus();
      });

      solapas.appendChild(boton);
      botones.push(boton);
      paneles.push(panel);
    });

    raiz.appendChild(solapas);
    paneles.forEach(function (p) { raiz.appendChild(p); });
    raiz.appendChild(crear("div", "grupo-nota-ref",
      "La diferencia es contra el promedio nacional, " + nacionalR + "%."));
  }

  // ---------------------------------------------------------------------
  // Arranque
  // ---------------------------------------------------------------------

  function versionPedida() {
    var q = new URLSearchParams(window.location.search);
    return M.version(q.getAll("resumen"), q.getAll("apaisado"));
  }

  /**
   * La versión resumida se acomoda sola según el ancho: en dos columnas en
   * escritorio y en caja vertical en móvil. Un solo embed para toda la home.
   *
   * En el HTML la tarjeta vive dentro de la banda gris, debajo de la barra: es
   * el orden de la caja. En dos columnas se la saca de la banda y se la pone al
   * lado del formulario; al volver a caja, se la devuelve EXACTAMENTE a su
   * lugar —después de la barra—, así que la caja en móvil queda idéntica a la
   * de siempre, no parecida. Se reordena el DOM en vez de usar sólo CSS porque
   * la tarjeta y los campos no son hermanos, y `display: contents` sobre la
   * banda le borraba el fondo gris.
   *
   * Reacciona a cambios de ancho —girar el teléfono, achicar la ventana— y no
   * sólo al arrancar.
   */
  function acomodarResumen() {
    var raiz = document.getElementById("widget");
    var campos = document.getElementById("campos");
    var tarjeta = raiz.querySelector(".result-card");
    var barra = raiz.querySelector(".prob-bar-wrapper");

    // La fila se crea una sola vez, vacía; los nodos entran y salen de ella.
    var fila = document.createElement("div");
    fila.className = "fila-apaisada";
    campos.parentNode.insertBefore(fila, campos);

    var actual = null;
    function aplicar() {
      var modo = M.disposicion(raiz.parentNode.getBoundingClientRect().width);
      if (modo === actual) return;
      actual = modo;
      // Mover un nodo en el DOM le saca el foco a lo que tenga adentro: quien
      // estaba en un desplegable cuando el iframe cruzó el corte —al rotar la
      // tablet, al achicar la ventana— quedaba sin foco. Se guarda y se
      // devuelve. Lo marcó Codex.
      var conFoco = document.activeElement;
      if (modo === "columnas") {
        raiz.classList.add("apaisado");
        fila.appendChild(campos);
        fila.appendChild(tarjeta);
      } else {
        raiz.classList.remove("apaisado");
        fila.parentNode.insertBefore(campos, fila);
        barra.parentNode.insertBefore(tarjeta, barra.nextSibling);
      }
      if (conFoco && conFoco !== document.activeElement && raiz.contains(conFoco)) {
        conFoco.focus({ preventScroll: true });
      }
    }
    aplicar();
    window.addEventListener("resize", aplicar);
  }

  /**
   * Si alguna opción elegida no entra en su desplegable, se achica la letra
   * de TODOS los desplegables, parejo.
   *
   * En celular los campos van de a dos y cada desplegable mide media columna:
   * "Terciaria completa o más" o "Delgado (Coalición)" no entraban y el
   * lector veía la opción elegida cortada. Ponerlos en una fila entera
   * alargaba la caja de la home (Tomer, 21/9/2026) y abreviarlos quedaba feo.
   * Achicar sólo el desplegable que no entraba dejaba letras de tamaños
   * distintos lado a lado; por prolijidad bajan todos al mismo tamaño: el que
   * necesita la opción elegida más larga. Donde todo entra —escritorio, dos
   * columnas— no cambia nada.
   *
   * Hay un piso de 11px: más chico no se lee. A 360px, con "Terciaria completa
   * o más" elegida, quedan en ~11px; a 320 puede quedar algo cortada igual.
   */
  var LETRA_MINIMA = 11;
  function ajustarLetraDesplegables() {
    var lienzo = document.createElement("canvas").getContext("2d");
    var selects = [].slice.call(document.querySelectorAll("#campos select"));
    function ajustar() {
      selects.forEach(function (sel) { sel.style.fontSize = ""; });
      var letra = Infinity;
      selects.forEach(function (sel) {
        var cs = getComputedStyle(sel);
        var base = parseFloat(cs.fontSize);
        var util = sel.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
        lienzo.font = cs.fontWeight + " " + base + "px " + cs.fontFamily;
        var ancho = lienzo.measureText(sel.options[sel.selectedIndex].text).width;
        var entra = ancho > util && util > 0 ? Math.floor(base * util / ancho * 10) / 10 : base;
        letra = Math.min(letra, entra);
      });
      var base = parseFloat(getComputedStyle(selects[0]).fontSize);
      if (letra < base) {
        letra = Math.max(LETRA_MINIMA, letra) + "px";
        selects.forEach(function (sel) { sel.style.fontSize = letra; });
      }
    }
    selects.forEach(function (sel) { sel.addEventListener("change", ajustar); });
    ajustar();
    window.addEventListener("resize", ajustar);
    // La medida depende de la fuente: si todavía no cargó, se mide con la de
    // reemplazo y hay que volver a medir cuando llega.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(ajustar);
  }

  function iniciar(modelo) {
    var v = versionPedida();
    var resumen = v.resumen;
    // POR ID, no por clase: el aviso de error y el bloque de <noscript>
    // también son ".marco", y `querySelector` devolvía el primero — o sea
    // que se le sacaba el `hidden` al div de error, vacío, y el widget real
    // quedaba invisible con todo bien calculado adentro.
    document.getElementById("widget").hidden = false;

    if (resumen) {
      document.getElementById("bloque-comparacion").remove();
      document.getElementById("bloque-metodologia").remove();
      document.getElementById("pie-completo").remove();
    } else {
      document.getElementById("pie-caja").remove();
    }
    if (resumen) {
      // La clase engancha el bloque "más compacta" del CSS.
      document.getElementById("widget").classList.add("resumen");
      acomodarResumen();
    }

    var campos = document.getElementById("campos");
    var nodos = {
      indicador: document.querySelector(".prob-indicator"),
      pastilla: document.querySelector(".prob-label"),
      numero: document.querySelector(".result-number"),
      texto: document.querySelector(".result-text")
    };

    function actualizar() {
      pintarResultado(nodos, M.predecir(modelo, leerPerfil(campos, modelo.variable_ranges)));
    }

    construirCampos(campos, modelo.variable_ranges, actualizar);
    ajustarLetraDesplegables();
    actualizar();

    if (!resumen) {
      construirComparacion(document.getElementById("comparacion"), modelo);
      document.getElementById("n-observaciones").textContent =
        modelo.model_info.n_observations;
      document.getElementById("pseudo-r2").textContent =
        (modelo.model_info.pseudo_r2 * 100).toFixed(1).replace(".", ",") + "%";
    }

    vigilarAltura();
  }

  /**
   * Le dice al que nos embebe cuánto medimos, para que ajuste el iframe.
   *
   * Es lo que hace que el embebido NO necesite una altura cableada: el script
   * de la nota escucha este mensaje. Mismo mecanismo que usan Flourish y
   * Datawrapper, y la razón por la que ellos no te piden un `height`.
   *
   * SE MIDE EL CONTENIDO, NO EL DOCUMENTO, y la diferencia importa: dentro de
   * un iframe el `html` y el `body` se estiran hasta el alto del iframe, así
   * que medirlos devuelve el alto que el iframe YA tiene. Eso arma un bucle:
   * el widget avisa el alto que le dieron, el padre se lo confirma, y el
   * iframe se queda clavado en su altura inicial — medido, los dos widgets de
   * la prueba quedaban en los 1860px de arranque.
   *
   * El borde inferior de `#widget` sí es el final del contenido. Se le suma el
   * margen negativo con el que la banda sangra hasta el borde del marco.
   */
  function avisarAltura() {
    if (window.parent === window) return;
    var raiz = document.getElementById("widget");
    if (!raiz) return;
    var rect = raiz.getBoundingClientRect();
    var alto = Math.ceil(rect.bottom + window.scrollY);
    window.parent.postMessage({ tipo: "ive-widget:alto", alto: alto, id: window.name || null }, "*");
  }

  /**
   * Avisa la altura cada vez que cambia de verdad, no sólo al arrancar.
   *
   * Hacía falta: el primer aviso salía ANTES de que terminaran de bajar las
   * fuentes de Google, y con la tipografía definitiva el texto ocupa más
   * líneas — medido, el iframe quedaba entre 5 y 18px corto y recortaba el
   * pie. Un `ResizeObserver` cubre eso y también lo que cambia después: abrir
   * la metodología, cambiar de solapa o girar el teléfono.
   */
  function vigilarAltura() {
    avisarAltura();
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(avisarAltura).observe(document.documentElement);
    }
    window.addEventListener("resize", avisarAltura);
    // Cinturón y tirantes para los navegadores sin ResizeObserver.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(avisarAltura);
    [120, 400, 1200].forEach(function (ms) { setTimeout(avisarAltura, ms); });
  }

  fetch("modelo.json")
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(iniciar)
    .catch(function (e) {
      var aviso = document.getElementById("error");
      aviso.hidden = false;
      aviso.textContent = "No se pudo cargar el modelo (" + e.message + ").";
    });

})();
