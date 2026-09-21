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

  // Mapeos UI → código del modelo. Son los mismos que en widgets/ive/config.py
  // y components.py; los índices arrancan en 1 como en el Python.
  var BALOTAJE_UI_A_CODIGO = {
    "No votó/Blanco": "otros",
    "Orsi (FA)": "martinez",
    "Delgado (Coalición)": "lacalle"
  };

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

      // El valor por defecto: en las variables numéricas `default` es el
      // código (base 1) y en las binarias es el índice; en balotaje es la
      // cadena del código. Se normaliza al índice de la lista.
      var indice = 0;
      if (campo[2] === "balotaje") {
        rango.labels.forEach(function (texto, i) {
          if (BALOTAJE_UI_A_CODIGO[texto] === rango.default) indice = i;
        });
      } else if (rango.default !== undefined) {
        indice = rango.options.indexOf(rango.default);
        if (indice < 0) indice = 0;
      }
      sel.value = String(indice);

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
    return {
      // Los códigos numéricos arrancan en 1, como en el Python.
      tramoEdad: idx("tramoEdad") + 1,
      esMujer: idx("esMujer"),
      nivelEduc: idx("nivelEduc") + 1,
      religiosidad: idx("religiosidad") + 1,
      esMontevideo: idx("esMontevideo"),
      tieneHijos: idx("tieneHijos"),
      hogar: idx("hogar") + 1,
      balotaje: BALOTAJE_UI_A_CODIGO[rangos.balotaje.labels[idx("balotaje")]]
    };
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
    var paneles = [];

    dimensiones.forEach(function (dim, i) {
      var boton = crear("button", "solapa", dim[0]);
      boton.type = "button";
      boton.setAttribute("role", "tab");
      boton.setAttribute("aria-selected", i === 0 ? "true" : "false");

      var panel = crear("div", "grupo-cifras");
      panel.setAttribute("role", "tabpanel");
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

      boton.addEventListener("click", function () {
        solapas.querySelectorAll(".solapa").forEach(function (b) {
          b.setAttribute("aria-selected", "false");
        });
        paneles.forEach(function (p) { p.hidden = true; });
        boton.setAttribute("aria-selected", "true");
        panel.hidden = false;
      });

      solapas.appendChild(boton);
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

  function esResumen() {
    var v = new URLSearchParams(window.location.search).getAll("resumen");
    // Con el parámetro repetido gana el ÚLTIMO, igual que en la versión
    // Streamlit (que sigue la semántica de st.query_params).
    var ultimo = v.length ? v[v.length - 1] : null;
    return ["1", "true", "si", "sí"].indexOf(String(ultimo).trim().toLowerCase()) !== -1;
  }

  function iniciar(modelo) {
    var resumen = esResumen();
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
    actualizar();

    if (!resumen) {
      construirComparacion(document.getElementById("comparacion"), modelo);
      document.getElementById("n-observaciones").textContent =
        modelo.model_info.n_observations;
      document.getElementById("pseudo-r2").textContent =
        (modelo.model_info.pseudo_r2 * 100).toFixed(1).replace(".", ",") + "%";
    }

    avisarAltura();
    window.addEventListener("resize", avisarAltura);
    // El alto cambia al abrir la metodología o cambiar de solapa.
    document.addEventListener("click", function () { setTimeout(avisarAltura, 60); });
  }

  /**
   * Le dice al que nos embebe cuánto medimos, para que ajuste el iframe.
   *
   * Es lo que hace que el embebido NO necesite una altura cableada: el script
   * de la nota escucha este mensaje. Es el mismo mecanismo que usan Flourish y
   * Datawrapper, y es la razón por la que ellos no te piden un `height`.
   */
  function avisarAltura() {
    if (window.parent === window) return;
    var alto = Math.ceil(document.documentElement.getBoundingClientRect().height);
    window.parent.postMessage({ tipo: "ive-widget:alto", alto: alto, id: window.name || null }, "*");
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
