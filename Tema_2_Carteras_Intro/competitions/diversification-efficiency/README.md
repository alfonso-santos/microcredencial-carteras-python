# Competicion de diversificacion

Paquete local para alumnos y profesor. No modifica la API ni utiliza servicios de nube.

## Abrir los materiales

`dist/latest.json` indica las rutas de la ultima edicion construida.
Dentro de `dist/<competition_id>/`:

- `alumnos/competicion_alumnos.ipynb`: enunciado, datos y comprobacion de una propuesta. Sin soluciones.
- `profesor/evaluacion_profesor.ipynb`: evaluacion por tickers, registro persistente, notas y clasificacion.
- `alumnos.zip`: unico paquete que se entrega a los alumnos.
- `profesor.zip`: paquete privado con el evaluador y la evidencia de calibracion.

Conserva cada notebook junto con sus carpetas auxiliares. Ejecuta su kernel desde esa carpeta.
El notebook del profesor necesita `challenge.py` a su lado.

## Uso del profesor

1. Abre `evaluacion_profesor.ipynb` y ejecuta todas las celdas. No se crean alumnos ficticios.
2. Introduce `student_id`, `public_alias` y `submitted_tickers` en la celda de registro.
3. Ejecuta esa celda. La nota y la tabla se actualizan automaticamente.
4. Tambien puedes llamar a `register_student("ID", "Alias", "TICKER1, TICKER2, ...")`.
5. Consulta `public/clasificacion.html` para ordenar, filtrar y compartir la clasificacion.

Los tickers se comprueban contra el universo oficial, deben ser distintos y estar entre 5 y 30.
Una entrega duplicada no se registra otra vez. Se conserva la mejor cartera por identificador.
Una propuesta que incumple el umbral se guarda como no clasificada, con nota tecnica cero.
Las propuestas con formato, universo o cardinalidad invalidos no se guardan.
Los empates exactos en K y volatilidad comparten posicion; las diferencias no se deciden con valores redondeados.

El formulario grafico es opcional y requiere `ipywidgets`. Sin esa libreria funciona la celda de registro.
El archivo JSON privado es para un solo kernel escritor; no es una base de datos multiusuario.
Haz copia de `private/submissions.json` antes de operaciones administrativas.
No mezcles registros de versiones diferentes de la competicion.

## Publicar la tabla

El HTML es autonomo: no requiere servidor ni librerias externas para buscar y ordenar.
Contiene alias, posicion, numero de acciones, volatilidad, puntuacion, nota tecnica y estado.
No contiene tickers, correos ni identificadores privados.

Registrar una entrega regenera el HTML local. Recarga la pagina para ver la nueva version.
Puedes adjuntarlo al campus o reemplazarlo en un alojamiento autorizado.
No hay publicacion automatica en internet ni actualizacion remota entre ordenadores.
No distribuyas la carpeta del profesor: incluye las soluciones.

## Reglas y nota

La muestra oficial elimina activos con mas de 20% de precios ausentes y despues fechas incompletas.
Se mantienen pesos iguales, log-retornos, desviacion muestral y anualizacion por raiz de 252.
El margen sobre la referencia es 0,25% relativo. Todo se fija en `data/competition.json`.

Entre las carteras clasificadas, primero gana el menor K y despues la menor volatilidad.
Puntuacion tecnica: `clip(100 * (Kmax - K) / (Kmax - Kref), 0, 100)`.
Nota tecnica: puntos divididos por diez.
Una cartera admisible de Kmax acciones recibe cero puntos: ha cumplido el limite pero no el objetivo de reducir posiciones.
Si Kref coincide con Kmax, todas las carteras admisibles reciben 100 puntos y se ordenan por K y volatilidad.
La defensa individual y la comprension se valoran aparte; la nota automatica solo mide el resultado tecnico.

Si `reference_status` es `provisional`, Kref es el menor tamano encontrado, NO un minimo demostrado.
Una cartera admisible con menos acciones obtiene 100 puntos, una posicion mejor y una alerta al profesor.
No se modifica automaticamente la escala de una edicion que ya esta en marcha.
Si se exige una escala basada exclusivamente en optimos certificados, no utilizar estas notas como definitivas hasta cerrar la certificacion.

## Calcular las referencias

`prepare_reference.py` reproduce el notebook de soluciones existente y genera datos y candidatos en `build/`.
`optimize_reference.py` resuelve un problema entero-cuadratico independiente para cada K.
Acepta tres formulaciones equivalentes en puntos binarios: directa, factorizada y redistribucion diagonal.
La ultima conserva una parte cuadratica semidefinida positiva y expone el resto como termino lineal.
No se limita el universo a las acciones de una heuristica.

```powershell
python -m pip install -r requirements-solver.txt
python prepare_reference.py
python optimize_reference.py --formulation diagonal --seconds-per-k 300 --output optimizer_long.json
python build_materials.py
python verify_materials.py
```

Tambien se puede concentrar el esfuerzo en tamanos concretos:

```powershell
python optimize_reference.py --sizes 18 19 22 --seconds-per-k 900 --output optimizer_focused.json
```

Cada ejecucion guarda un informe distinto. `build_materials.py` combina las mejores cotas compatibles por K.
Una cota superior procede de una cartera real. La cota inferior permite acotar lo que podria quedar por mejorar.
`timelimit` no es un certificado. Un certificado numerico depende de las tolerancias del solver.
La brecha se mide en varianza, no directamente en volatilidad.

La referencia global esta certificada cuando su brecha relativa de varianza es como maximo `1e-6`.
La cardinalidad minima esta certificada cuando todos los tamanos menores tienen cota inferior de volatilidad mayor al umbral.
El informe conserva ambos estados por separado.

Los calculos locales pueden tardar bastante. No se promete cerrar todas las brechas en un tiempo prefijado.
Congela una edicion antes de recibir entregas. Una recalibracion genera otro identificador si cambian las reglas.
No reconstruyas materiales de una clase en marcha para cambiar sus referencias.

## Verificacion

```powershell
python -m unittest -v test_challenge.py
python -m unittest -v test_optimizer.py
```

Se prueban validacion, nota, empates, duplicados, historial, privacidad del HTML y cambio de version.
Las formulaciones del optimizador se contrastan con enumeracion exhaustiva en un universo sintetico pequeno.
Los notebooks se ejecutan y validan por separado. Los ejemplos de pruebas no contaminan el registro del profesor.

## Riesgos metodologicos

Es una competicion dentro de muestra, no un backtest predictivo.
El filtro de disponibilidad sobre toda la historia no seria apropiado para una estrategia historica sin adaptar su tratamiento temporal.
No se ha auditado la composicion historica del universo ni el ajuste de precios por eventos corporativos.
La media de log-retornos es la convencion de puntuacion, no el retorno simple exacto de una cartera rebalanceada.
La identidad y el orden del dataset se verifican mediante SHA256.

## Fuentes tecnicas

- [Instalacion oficial de PySCIPOpt](https://pyscipopt.readthedocs.io/en/stable/install.html)
- [Modelo, estados y parametros](https://pyscipopt.readthedocs.io/en/latest/tutorials/model.html)
- [API del optimizador y cotas](https://pyscipopt.readthedocs.io/en/stable/api/model.html)
