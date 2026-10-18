import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

TOL = 1e-7

VARIABLES = {
    "v0": ("Velocidad inicial", "m/s"),
    "vf": ("Velocidad final", "m/s"),
    "a": ("Aceleración", "m/s²"),
    "t": ("Tiempo", "s"),
    "d": ("Desplazamiento", "m"),
}

COLORES = {
    "fondo": "#F3F6FC",
    "azul": "#173B6C",
    "azul_claro": "#2F75B5",
    "blanco": "#FFFFFF",
    "texto": "#243247",
    "verde": "#21865B",
    "rojo": "#B42318",
    "borde": "#D6DFEC",
}


# ==========================================================
# FUNCIONES MATEMÁTICAS
# ==========================================================

def casi_iguales(x, y):
    return math.isclose(
        x, y,
        rel_tol=TOL,
        abs_tol=TOL
    )


def leer_numero(texto):
    texto = texto.strip()

    if not texto:
        return None

    valor = float(texto.replace(",", "."))

    if not math.isfinite(valor):
        raise ValueError("Los valores deben ser números finitos.")

    return valor


def validar_datos(datos):
    errores = []

    v0 = datos["v0"]
    vf = datos["vf"]
    a = datos["a"]
    t = datos["t"]
    d = datos["d"]

    if t is not None and t < 0:
        errores.append("El tiempo no puede ser negativo.")

    # Ecuación: vf = v0 + at
    if all(x is not None for x in (v0, vf, a, t)):
        esperado = v0 + a * t

        if not casi_iguales(vf, esperado):
            errores.append(
                f"La velocidad final debería ser "
                f"{esperado:.4f} m/s."
            )

    # Ecuación: d = v0t + at²/2
    if all(x is not None for x in (v0, a, t, d)):
        esperado = v0 * t + 0.5 * a * t**2

        if not casi_iguales(d, esperado):
            errores.append(
                f"El desplazamiento debería ser "
                f"{esperado:.4f} m."
            )

    # Ecuación: vf² = v0² + 2ad
    if all(x is not None for x in (v0, vf, a, d)):
        esperado = v0**2 + 2 * a * d

        if esperado < -TOL:
            errores.append(
                "Los datos implican una velocidad final "
                "al cuadrado negativa."
            )
        elif not casi_iguales(vf**2, max(0.0, esperado)):
            errores.append(
                "Los datos no cumplen vf² = v0² + 2ad."
            )

    # Ecuación: d = ((v0 + vf)/2)t
    if all(x is not None for x in (v0, vf, t, d)):
        esperado = (v0 + vf) * t / 2

        if not casi_iguales(d, esperado):
            errores.append(
                "Los datos no cumplen "
                "d = ((v0 + vf)/2)t."
            )

    return list(dict.fromkeys(errores))


def resolver_mruv(datos):
    """
    Calcula variables desconocidas mediante las ecuaciones
    del MRUV. Si existen dos tiempos posibles, los informa.
    """

    r = datos.copy()
    avisos = []
    tiempos_posibles = []

    errores = validar_datos(datos)

    if errores:
        return r, errores, tiempos_posibles

    v0 = r["v0"]
    vf = r["vf"]
    a = r["a"]
    t = r["t"]
    d = r["d"]

    # Resolver tiempo mediante la ecuación cuadrática:
    # d = v0*t + (a*t²)/2
    if t is None and all(x is not None for x in (v0, a, d)):

        if casi_iguales(a, 0):
            if not casi_iguales(v0, 0):
                candidato = d / v0

                if candidato >= -TOL:
                    tiempos_posibles = [max(0.0, candidato)]

            elif casi_iguales(d, 0):
                avisos.append(
                    "El tiempo no puede determinarse de forma única "
                    "con v0 = 0, a = 0 y d = 0."
                )

        else:
            discriminante = v0**2 + 2 * a * d

            if discriminante < -TOL:
                avisos.append(
                    "No existen tiempos reales para los datos dados."
                )
            else:
                raiz = math.sqrt(max(0.0, discriminante))

                raices = [
                    (-v0 + raiz) / a,
                    (-v0 - raiz) / a
                ]

                tiempos_posibles = sorted({
                    round(max(0.0, x), 10)
                    for x in raices
                    if x >= -TOL
                })

        if len(tiempos_posibles) == 1:
            r["t"] = tiempos_posibles[0]

        elif len(tiempos_posibles) == 2:
            avisos.append(
                "Existen dos tiempos posibles: "
                + " s y ".join(
                    f"{x:.4f}" for x in tiempos_posibles
                )
                + " s. Introduce una condición adicional "
                  "para elegir uno."
            )

    # Resolver las variables que se puedan determinar.
    for _ in range(10):
        anterior = r.copy()

        v0 = r["v0"]
        vf = r["vf"]
        a = r["a"]
        t = r["t"]
        d = r["d"]

        # vf = v0 + at
        if v0 is not None and a is not None and t is not None:
            if vf is None:
                r["vf"] = v0 + a * t

        # d = v0t + at²/2
        if v0 is not None and a is not None and t is not None:
            if d is None:
                r["d"] = v0 * t + 0.5 * a * t**2

        # v0 = vf - at
        if vf is not None and a is not None and t is not None:
            if v0 is None:
                r["v0"] = vf - a * t

        # a = (vf - v0)/t
        if v0 is not None and vf is not None and t is not None:
            if a is None and not casi_iguales(t, 0):
                r["a"] = (vf - v0) / t

        # a = 2(d - v0t)/t²
        if v0 is not None and d is not None and t is not None:
            if a is None and not casi_iguales(t, 0):
                r["a"] = 2 * (d - v0 * t) / t**2

        # v0 = (d - at²/2)/t
        if a is not None and d is not None and t is not None:
            if v0 is None and not casi_iguales(t, 0):
                r["v0"] = (d - 0.5 * a * t**2) / t

        # a = (vf² - v0²)/(2d)
        if v0 is not None and vf is not None and d is not None:
            if a is None and not casi_iguales(d, 0):
                r["a"] = (vf**2 - v0**2) / (2 * d)

        # t = (vf - v0)/a
        if v0 is not None and vf is not None and a is not None:
            if t is None and not casi_iguales(a, 0):
                candidato = (vf - v0) / a

                if candidato >= -TOL:
                    r["t"] = max(0.0, candidato)

        # d = ((v0 + vf)/2)t
        if v0 is not None and vf is not None and t is not None:
            if d is None:
                r["d"] = (v0 + vf) * t / 2

        # t = 2d/(v0 + vf)
        if v0 is not None and vf is not None and d is not None:
            if t is None:
                vm_doble = v0 + vf

                if not casi_iguales(vm_doble, 0):
                    candidato = 2 * d / vm_doble

                    if candidato >= -TOL:
                        r["t"] = max(0.0, candidato)

        if r == anterior:
            break

    # Comprobar consistencia de los resultados.
    errores_finales = validar_datos(r)

    if errores_finales:
        avisos.extend(errores_finales)

    pendientes = [
        VARIABLES[k][0]
        for k in VARIABLES
        if r[k] is None
    ]

    if pendientes:
        avisos.append(
            "Variables no determinadas: "
            + ", ".join(pendientes)
            + ". Introduce más datos."
        )

    return r, list(dict.fromkeys(avisos)), tiempos_posibles


# ==========================================================
# INTERFAZ: OBTENER Y MOSTRAR DATOS
# ==========================================================

def obtener_datos():
    datos = {}
<<<<<<< HEAD

    for clave, campo in campos.items():
        try:
            datos[clave] = leer_numero(campo.get())
        except ValueError:
            raise ValueError(
                f"El campo '{VARIABLES[clave][0]}' "
                "contiene un número inválido."
            )

    return datos


def mostrar_resultados(resultados, originales):
    salida = []

    for clave, (nombre, unidad) in VARIABLES.items():
        valor = resultados[clave]

        if valor is None:
            estado = "No determinado"
            salida.append(f"{nombre}: {estado}")

        else:
            if originales[clave] is not None:
                estado = "Ingresado"
            else:
                estado = "Calculado"

            salida.append(
                f"{nombre}: {valor:.4f} {unidad} ({estado})"
            )

    caja_resultados.config(state="normal")
    caja_resultados.delete("1.0", tk.END)
    caja_resultados.insert(tk.END, "\n".join(salida))
    caja_resultados.config(state="disabled")


def calcular():
    try:
        datos = obtener_datos()
    except ValueError as error:
        messagebox.showerror("Error de entrada", str(error))
        return

    cantidad = sum(v is not None for v in datos.values())

    if cantidad < 3:
        messagebox.showwarning(
            "Datos insuficientes",
            "Introduce al menos tres variables conocidas."
        )
        return

    errores = validar_datos(datos)

    if errores:
        messagebox.showerror(
            "Datos incompatibles",
            "\n\n".join(errores)
        )
        return

    resultados, avisos, tiempos = resolver_mruv(datos)

    mostrar_resultados(resultados, datos)

    if avisos:
        messagebox.showwarning(
            "Información del cálculo",
            "\n\n".join(avisos)
        )

    if all(
        resultados[k] is not None
        for k in ("v0", "a", "t")
    ):
        graficar(
            resultados["v0"],
            resultados["a"],
            resultados["t"]
        )
    else:
        limpiar_grafica()


# ==========================================================
# GRÁFICAS
# ==========================================================


def graficar(v0, a, t_final):
    limpiar_grafica()

    if t_final < 0:
        return

    tiempos = (
        np.linspace(0, t_final, 200)
        if t_final > 0
        else np.array([0.0])
    )

    # Cálculos del movimiento
    posicion = v0 * tiempos + 0.5 * a * tiempos**2
    velocidad = v0 + a * tiempos
    aceleracion = np.full_like(tiempos, a)

    # Crear una figura más compacta y adaptable
    figura = Figure(figsize=(10, 5), dpi=100)
    figura.patch.set_facecolor(COLORES["blanco"])

    e1, e2, e3 = figura.subplots(3, 1, sharex=True)

    # Gráfica 1: posición o desplazamiento
    e1.plot(
        tiempos, posicion,
        color=COLORES["azul_claro"],
        linewidth=2,
        label="Desplazamiento"
    )
    e1.set_title("Desplazamiento frente al tiempo", fontsize=10, pad=8)
    e1.set_ylabel("Desplazamiento (m)", fontsize=9)
    e1.legend(loc="upper left", fontsize=8)
    e1.grid(True, alpha=0.3)

    # Gráfica 2: velocidad
    e2.plot(
        tiempos, velocidad,
        color="#21865B",
        linewidth=2,
        label="Velocidad"
    )
    e2.set_title("Velocidad frente al tiempo", fontsize=10, pad=8)
    e2.set_ylabel("Velocidad (m/s)", fontsize=9)
    e2.legend(loc="upper left", fontsize=8)
    e2.grid(True, alpha=0.3)

    # Gráfica 3: aceleración
    e3.plot(
        tiempos, aceleracion,
        color="#B42318",
        linewidth=2,
        label="Aceleración"
    )
    e3.set_title("Aceleración frente al tiempo", fontsize=10, pad=8)
    e3.set_xlabel("Tiempo (s)", fontsize=9)
    e3.set_ylabel("Aceleración (m/s²)", fontsize=9)
    e3.legend(loc="upper left", fontsize=8)
    e3.grid(True, alpha=0.3)

    # Mejorar la separación entre gráficas y etiquetas
    
    
    # Mejorar la separación entre gráficas y etiquetas
    for eje in (e1, e2, e3):
        eje.set_facecolor("#FAFCFF")
        eje.tick_params(axis="both", labelsize=8)
        eje.margins(x=0.02)

        # Mostrar los números debajo del eje X
        eje.tick_params(
            axis="x",
            labelbottom=True,
            labeltop=False
        )

    figura.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.95,
        bottom=0.12,
        hspace=0.85
    )


    # Insertar la figura en Tkinter
    global lienzo_actual

    lienzo_actual = FigureCanvasTkAgg(
        figura,
        master=marco_grafica
    )

    lienzo_actual.draw()

    lienzo_actual.get_tk_widget().pack(
        fill="both",
        expand=True
    )



def limpiar_grafica():
    global lienzo_actual

    if lienzo_actual is not None:
        lienzo_actual.get_tk_widget().destroy()
        lienzo_actual = None


# ==========================================================
# BOTONES
# ==========================================================

def limpiar():
    for campo in campos.values():
        campo.delete(0, tk.END)

    caja_resultados.config(state="normal")
    caja_resultados.delete("1.0", tk.END)
    caja_resultados.insert(
        tk.END,
        "Introduce los datos y pulsa Calcular."
    )
    caja_resultados.config(state="disabled")

    limpiar_grafica()


def cargar_ejemplo():
    limpiar()

    campos["v0"].insert(0, "4")
    campos["a"].insert(0, "2")
    campos["t"].insert(0, "5")

=======

    for clave, campo in campos.items():
        try:
            datos[clave] = leer_numero(campo.get())
        except ValueError:
            raise ValueError(
                f"El campo '{VARIABLES[clave][0]}' "
                "contiene un número inválido."
            )

    return datos


def mostrar_resultados(resultados, originales):
    salida = []

    for clave, (nombre, unidad) in VARIABLES.items():
        valor = resultados[clave]

        if valor is None:
            estado = "No determinado"
            salida.append(f"{nombre}: {estado}")

        else:
            if originales[clave] is not None:
                estado = "Ingresado"
            else:
                estado = "Calculado"

            salida.append(
                f"{nombre}: {valor:.4f} {unidad} ({estado})"
            )

    caja_resultados.config(state="normal")
    caja_resultados.delete("1.0", tk.END)
    caja_resultados.insert(tk.END, "\n".join(salida))
    caja_resultados.config(state="disabled")


def calcular():
    try:
        datos = obtener_datos()
    except ValueError as error:
        messagebox.showerror("Error de entrada", str(error))
        return

    cantidad = sum(v is not None for v in datos.values())

    if cantidad < 3:
        messagebox.showwarning(
            "Datos insuficientes",
            "Introduce al menos tres variables conocidas."
        )
        return

    errores = validar_datos(datos)

    if errores:
        messagebox.showerror(
            "Datos incompatibles",
            "\n\n".join(errores)
        )
        return

    resultados, avisos, tiempos = resolver_mruv(datos)

    mostrar_resultados(resultados, datos)

    if avisos:
        messagebox.showwarning(
            "Información del cálculo",
            "\n\n".join(avisos)
        )

    if all(
        resultados[k] is not None
        for k in ("v0", "a", "t")
    ):
        graficar(
            resultados["v0"],
            resultados["a"],
            resultados["t"]
        )
    else:
        limpiar_grafica()


# ==========================================================
# GRÁFICAS
# ==========================================================

def graficar(v0, a, t_final):
    limpiar_grafica()

    if t_final < 0:
        return

    tiempos = (
        np.linspace(0, t_final, 200)
        if t_final > 0
        else np.array([0.0])
    )

    # Desplazamiento desde la posición inicial.
    posicion = v0 * tiempos + 0.5 * a * tiempos**2

    velocidad = v0 + a * tiempos
    aceleracion = np.full_like(tiempos, a)

    figura = Figure(figsize=(8, 7), dpi=100)
    figura.patch.set_facecolor(COLORES["blanco"])

    e1 = figura.add_subplot(311)
    e2 = figura.add_subplot(312)
    e3 = figura.add_subplot(313)

    e1.plot(tiempos, posicion, linewidth=2, label="Desplazamiento")
    e1.set_title("Desplazamiento frente al tiempo")
    e1.set_ylabel("Desplazamiento (m)")
    e1.grid(True, alpha=0.3)
    e1.legend()

    e2.plot(tiempos, velocidad, linewidth=2, label="Velocidad")
    e2.set_title("Velocidad frente al tiempo")
    e2.set_ylabel("Velocidad (m/s)")
    e2.grid(True, alpha=0.3)
    e2.legend()

    e3.plot(tiempos, aceleracion, linewidth=2, label="Aceleración")
    e3.set_title("Aceleración frente al tiempo")
    e3.set_xlabel("Tiempo (s)")
    e3.set_ylabel("Aceleración (m/s²)")
    e3.grid(True, alpha=0.3)
    e3.legend()

    for eje in (e1, e2, e3):
        eje.set_xlim(left=0)
        eje.set_facecolor("#FAFCFF")

    figura.tight_layout()

    global lienzo_actual

    lienzo_actual = FigureCanvasTkAgg(
        figura,
        master=marco_grafica
    )

    lienzo_actual.draw()

    lienzo_actual.get_tk_widget().pack(
        fill="both",
        expand=True
    )


def limpiar_grafica():
    global lienzo_actual

    if lienzo_actual is not None:
        lienzo_actual.get_tk_widget().destroy()
        lienzo_actual = None


# ==========================================================
# BOTONES
# ==========================================================

def limpiar():
    for campo in campos.values():
        campo.delete(0, tk.END)

    caja_resultados.config(state="normal")
    caja_resultados.delete("1.0", tk.END)
    caja_resultados.insert(
        tk.END,
        "Introduce los datos y pulsa Calcular."
    )
    caja_resultados.config(state="disabled")

    limpiar_grafica()


def cargar_ejemplo():
    limpiar()

    campos["v0"].insert(0, "4")
    campos["a"].insert(0, "2")
    campos["t"].insert(0, "5")

>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
    calcular()


# ==========================================================
# DISEÑO PROFESIONAL
# ==========================================================

ventana = tk.Tk()
ventana.title("Simulador interactivo de MRUV")
<<<<<<< HEAD
ventana.geometry("1200x850")
ventana.minsize(1000, 750)
=======
ventana.geometry("1150x850")
ventana.minsize(900, 650)
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
ventana.configure(bg=COLORES["fondo"])

estilo = ttk.Style()
estilo.theme_use("clam")

estilo.configure(
    "TFrame",
    background=COLORES["fondo"]
)

estilo.configure(
    "TLabel",
    background=COLORES["fondo"],
    foreground=COLORES["texto"],
    font=("Arial", 10)
)

estilo.configure(
    "Titulo.TLabel",
    font=("Arial", 19, "bold"),
    foreground=COLORES["azul"]
)

estilo.configure(
    "Subtitulo.TLabel",
    font=("Arial", 10),
    foreground="#61718A"
)

estilo.configure(
    "TLabelframe",
    background=COLORES["blanco"],
    bordercolor=COLORES["borde"]
)

estilo.configure(
    "TLabelframe.Label",
    background=COLORES["blanco"],
    foreground=COLORES["azul"],
    font=("Arial", 10, "bold")
)

estilo.configure(
    "TButton",
    font=("Arial", 10, "bold"),
    padding=8
)

estilo.configure(
    "Principal.TButton",
    background=COLORES["azul"],
    foreground=COLORES["blanco"]
)

estilo.map(
    "Principal.TButton",
    background=[("active", COLORES["azul_claro"])]
)

estilo.configure(
    "TEntry",
    padding=6
)

contenedor = ttk.Frame(ventana, padding=15)
contenedor.pack(fill="both", expand=True)

ttk.Label(
    contenedor,
    text="SIMULADOR Y GRAFICADOR DE MRUV",
    style="Titulo.TLabel"
).pack(pady=(0, 4))

ttk.Label(
    contenedor,
    text="Movimiento Rectilíneo Uniformemente Variado",
    style="Subtitulo.TLabel"
).pack(pady=(0, 15))

panel = ttk.Frame(contenedor)
<<<<<<< HEAD
panel.pack(fill="both", expand=True)

panel.columnconfigure(0, weight=0)
panel.columnconfigure(1, weight=1)
panel.columnconfigure(2, weight=2)
panel.rowconfigure(0, weight=1)
=======
panel.pack(fill="x")
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d

marco_entrada = ttk.LabelFrame(
    panel,
    text="Datos del movimiento",
    padding=12
)

<<<<<<< HEAD
marco_entrada.grid(
    row=0, column=0,
    sticky="ns",
    padx=(0, 8)
=======
marco_entrada.pack(
    side="left",
    fill="y",
    padx=(0, 12)
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
)

campos = {}

for fila, (clave, (nombre, unidad)) in enumerate(VARIABLES.items()):
    ttk.Label(
        marco_entrada,
        text=f"{nombre} ({unidad})"
    ).grid(
        row=fila,
        column=0,
        sticky="w",
        padx=5,
        pady=7
    )

    campo = ttk.Entry(marco_entrada, width=18)

    campo.grid(
        row=fila,
        column=1,
        padx=5,
        pady=7
    )

    campos[clave] = campo

ttk.Label(
    marco_entrada,
    text="Deja vacías las variables desconocidas."
).grid(
    row=5,
    column=0,
    columnspan=2,
    sticky="w",
    padx=5,
    pady=8
)

ttk.Button(
    marco_entrada,
    text="Calcular y graficar",
    style="Principal.TButton",
    command=calcular
).grid(
    row=6,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=5,
    pady=5
)

ttk.Button(
    marco_entrada,
    text="Cargar ejemplo",
    command=cargar_ejemplo
).grid(
    row=7,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=5,
    pady=5
)

ttk.Button(
    marco_entrada,
    text="Limpiar todo",
    command=limpiar
).grid(
    row=8,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=5,
    pady=5
)

marco_resultados = ttk.LabelFrame(
    panel,
    text="Resultados del cálculo",
    padding=12
)

<<<<<<< HEAD
marco_resultados.grid(
    row=0, column=1,
    sticky="nsew",
    padx=8
=======
marco_resultados.pack(
    side="left",
    fill="both",
    expand=True
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
)

caja_resultados = tk.Text(
    marco_resultados,
<<<<<<< HEAD
    height=7,
=======
    height=13,
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
    wrap="word",
    font=("Consolas", 10),
    bg=COLORES["blanco"],
    fg=COLORES["texto"],
    relief="flat",
    padx=10,
    pady=10
)

caja_resultados.pack(fill="both", expand=True)

caja_resultados.insert(
    tk.END,
    "Introduce los datos y pulsa Calcular."
)

caja_resultados.config(state="disabled")

marco_grafica = ttk.LabelFrame(
<<<<<<< HEAD
    panel,
=======
    contenedor,
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
    text="Visualización gráfica del movimiento",
    padding=8
)

<<<<<<< HEAD
marco_grafica.grid(
    row=0,
    column=2,
    sticky="nsew",
    padx=(8, 0)
=======
marco_grafica.pack(
    fill="both",
    expand=True,
    pady=(15, 0)
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
)

lienzo_actual = None

<<<<<<< HEAD
ventana.mainloop()
=======
ventana.mainloop()
>>>>>>> b69ee6c72e5ac3b278cd076d297bb01d0c87b68d
