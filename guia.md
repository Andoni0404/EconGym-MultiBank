## La economía de partida

Partimos siempre de la economía de 1000 hogares y 50 empresas, con la deuda y los ahorros reales
de la encuesta SCF 2022. La de 100 hogares solo la usamos para programar y probar rápido, y como
primera lección del curriculum.

## Los tres niveles

Cada pregunta la hacemos en tres niveles, para ver cómo cambia la respuesta según el escenario:

- **Nivel 1, un banco solo.** No tiene competencia, es un monopolio.
- **Nivel 2, N bancos iguales.** Todos empiezan con los mismos clientes y el mismo capital.
- **Nivel 3, N bancos distintos.** Dos grandes, dos medianos y un neobanco que empieza con
  pocos clientes y poco capital. Es lo más parecido a un mercado de verdad: en España cinco bancos
  se reparten más o menos el 70% del mercado.

Ponemos cinco bancos y no dos porque un mercado de dos no se parece a nada real. Con dos bancos
solo hacemos pruebas rápidas para ver que el código funciona.

### La recompensa: ¿en qué nos basamos?

Es lo más importante de todo el trabajo, porque en RL el banco aprende exactamente lo que le
premiamos, ni más ni menos. Si la recompensa está mal pensada, el banco aprenderá algo raro aunque
el algoritmo funcione perfecto.

Para decidirla nos fijamos en qué busca un banco de verdad. Sus dueños quieren que el capital del
banco crezca (es lo que se llama rentabilidad sobre el capital, o ROE, y es el número que publican
todos los bancos), y el regulador no le deja quebrar. Así que la recompensa tiene que premiar que
crezca el capital y castigar la quiebra. Eso es lo que tenemos ahora, pero hay varias formas de
contar ese crecimiento, y no da igual cuál usemos:

- **Lo que crece el capital, dividido por el capital inicial** (lo que hay ahora). Es sencillo y se
  puede comparar entre bancos. El problema es que un banco que ya ha crecido mucho gana mucho más
  en euros, así que los últimos años del episodio pesan más que los primeros.
- **La rentabilidad de cada año** (lo que crece el capital dividido por el capital de ese año). Es
  justo el ROE que publican los bancos y todos los años cuentan igual. El problema es que, con muy
  poco capital después de una mala racha, la recompensa se dispara.
- **El crecimiento en logaritmo** (el logaritmo del capital de este año menos el del año anterior).
  Castiga una pérdida más de lo que premia una ganancia del mismo tamaño, así que el banco se vuelve
  prudente él solo. Y cuando el capital se acerca a cero, el logaritmo se va a menos infinito, así
  que quebrar ya es un desastre sin tener que añadirle un castigo aparte. Es la idea del criterio de
  Kelly, que se usa mucho en finanzas para no arruinarse.
- **La versión "aplastada"** (`sigmoide`, que ya está en el código). No cambia lo que se premia, solo
  hace que los números sean más suaves para que el entrenamiento sea más estable.

Mi propuesta es quedarnos con la primera como base, porque ya está hecha y probada, y en la
pregunta 2 compararla con la del logaritmo. Así respondemos una pregunta de RL muy interesante: ¿se
puede conseguir que el banco no quiebre solo eligiendo bien la forma de la recompensa, sin tener que

## Las cuatro preguntas

### 1. ¿Aprende el banco, y aprende antes si le enseñamos poco a poco?

Lo primero es ver si el banco aprende a hacerlo mejor que las reglas fijas. Aquí los rivales son
siempre reglas fijas, que no cambian nunca, así que el banco aprende contra algo quieto. Las reglas
son las de `multibank/reglas.py`: la prudente, la agresiva, la clásica y la competitiva, más la
"acción media", que es lo que haría una red sin entrenar.

Además probamos **curriculum learning**: en vez de soltar al banco directamente en la economía
completa, le enseñamos poco a poco. Primero en una economía sin impagos, luego con impagos pero sin
quiebra y al final con todo. Y también primero con 100 hogares y luego con 1000. Lo comparamos con
empezar de golpe con todo, y miramos cuál aprende antes y cuál acaba mejor. Si el curriculum
funciona, además nos ahorra tiempo de ordenador, porque la economía de 100 hogares va unas tres
veces más rápida.

- En el nivel 1: ¿le gana a las reglas un banco solo?
- En el nivel 2: ¿le sigue ganando cuando los otros cuatro son reglas que compiten con él?
- En el nivel 3: ¿aprende igual de bien siendo grande que siendo el neobanco?

### 2. ¿Cómo aprende cuando hay un desastre raro, la quiebra? (la recompensa)

Aquí es donde jugamos con la función de recompensa. Ahora mismo, cada año el banco recibe lo que ha
crecido su capital dividido por su capital inicial, y si quiebra se lleva además un castigo λ y se
acaba el episodio para él. Cambiando esa recompensa cambia lo que aprende, y eso es lo que miramos.

Primero comparamos dos formas de la recompensa: la normal, el cambio de capital tal cual, y una
"aplastada" (`recompensa="sigmoide"` en `config.py`), en la que un año muy bueno o muy malo no pesa
tanto. Así vemos si al banco le cambia la forma de arriesgar según cómo le contamos lo que gana.

Y luego probamos tres formas de enseñarle que quebrar es malo:

- **Tratarla bien o mal.** Bien es decirle al algoritmo que quebrar es el final de verdad
  (`terminated`); mal es tratarla como si el episodio simplemente se cortara (`truncated`).
  Esperamos que, tratada mal, el banco no le tenga miedo y quiebre más.
- **Castigarla más o menos.** Con castigos λ de 0, 0,5, 2 y 10.
- **Ponerle un límite** en vez de un castigo: "no quiebres más del 5% de las veces", y que el propio
  algoritmo ajuste el castigo solo (PPO-Lagrangiano, RL con restricciones).

Medimos cuántas veces quiebra mientras aprende y al final, y cuánto gana. Ahora mismo los bancos
casi nunca quiebran (entre un 5 y un 12% de las veces, aunque jueguen mal a propósito); si con cinco
bancos sigue igual, habrá que hacer la economía más dura para que esta pregunta tenga sentido.

- En el nivel 1: ¿cómo cambia lo que aprende según el castigo?
- En el nivel 2: con competencia, ¿arriesga más?
- En el nivel 3: ¿el neobanco, que tiene menos que perder, arriesga más que los grandes?

### 3. ¿Qué pasa cuando aprenden varios a la vez?

Aquí los rivales también están aprendiendo, así que lo que le funcionaba ayer a un banco hoy ya
no le funciona, porque el otro ha cambiado. Esto es lo difícil del RL multiagente.

Comparamos **IPPO** y **MAPPO**, pero no poniéndolos a pelear, sino viendo cuál aprende de forma
más estable. Con IPPO cada banco aprende mirando solo lo suyo; con MAPPO, mientras aprende, cada
banco tiene un entrenador (el crítico) que ve a todos y le ayuda a entender si le ha ido mal por su
culpa o por lo que han hecho los demás.

Además miramos qué comportamiento aparece solo: si los bancos acaban cobrando caro todos a la vez,
como si se pusieran de acuerdo sin hablar. Es lo que estudiaron Calvano y otros en 2020 con
algoritmos de RL que aprendían a poner precios. Para medirlo usamos su índice, que compara lo que
cobran nuestros bancos con lo que cobraría un monopolio y con lo que cobrarían compitiendo de verdad.

- En el nivel 1 no hay varios bancos, pero nos sirve de referencia: es el monopolio, lo máximo que
  se puede cobrar.
- En el nivel 2: ¿qué algoritmo aprende más estable? ¿Se ponen de acuerdo?
- En el nivel 3: ¿se ponen de acuerdo los grandes y dejan fuera al neobanco?

### 4. ¿Le sirve lo aprendido fuera de donde se entrenó?

Entrenamos con 1000 hogares y luego, sin volver a entrenar, ponemos al banco en una economía de
un millon de hogares y en economías con condiciones distintas (más o menos deuda, más o menos capital).
Entrenar con un millón de hogares no es posible, porque cada paso tardaría unos 12 segundos, pero
probar un banco ya entrenado sí, porque son solo unos pocos episodios.

Aquí probamos **domain randomization**: entrenar cada vez en una economía un poco distinta, con
más o menos deuda inicial, más o menos capital y más o menos empresas. Lo comparamos con entrenar
siempre en la misma economía, y miramos cuánto peor lo hace cada uno cuando le cambiamos el
escenario.

- En el nivel 1: ¿generaliza un banco solo?
- En el nivel 2: ¿y compitiendo?
- En el nivel 3: ¿y en un mercado con bancos distintos de los que vio al entrenar?

## Cómo lo hacemos para que salga bien

Que un algoritmo gane una vez puede ser pura casualidad: la semilla, cómo empezaron los pesos de la
red o el reparto de clientes. Por eso cada cosa se entrena con **5 semillas**, y no damos solo la
media, sino también cuánto varía. Si dos resultados se solapan, no decimos que uno es mejor que
otro. Y si no sale ninguna diferencia, lo decimos tal cual, porque eso también es un resultado.

La semilla decide dos cosas: cómo empiezan los pesos de la red y en qué economías se entrena el
banco. Cambiándola, el mismo algoritmo aprende cinco veces de cero, cada vez con otra suerte, y así
vemos si lo que sale es del algoritmo o de la suerte.

Las semillas son simplemente los números del 1 al 5. No hay que elegirlas "buenas": lo importante
es que estén fijadas antes de empezar y que no las cambiemos después. Lo que no vale es entrenar con
veinte semillas y quedarnos con las cinco que mejor salen, porque eso es hacer trampa. Lo que no
cambia nunca son las economías de la evaluación final: siempre son las mismas y distintas de las de
entrenamiento, para que todos los bancos hagan el mismo examen y ninguno lo haya visto antes.

¿Por qué cinco y no más? Es un equilibrio entre fiabilidad y tiempo. Con una o dos semillas no se
puede saber nada (Henderson y otros, 2018, enseñaron que el mismo algoritmo con semillas distintas
puede parecer dos algoritmos diferentes). Con tres ya se ve algo, pero cuánto varía el resultado sale
muy poco fiable. Cinco es lo mínimo que se suele pedir en RL, y con herramientas como *rliable*
(Agarwal y otros, 2021) ya se pueden sacar intervalos de confianza decentes. Diez sería mejor, pero
duplicaría las 100 horas de ordenador y no nos da tiempo. Si en algún caso dos resultados salen muy
pegados y no se puede decir cuál es mejor, ahí sí subimos a diez semillas solo para ese caso.


