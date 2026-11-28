# Diario de decisiones

Lo importante que hemos ido decidiendo y por qué, para acordarnos cuando toque presentar.
Lo más nuevo, arriba.

---

### Pendiente

- **La recompensa.** La idea es premiar al banco por lo que crece su capital cada año y castigarle
  fuerte si quiebra. Falta confirmarlo. Cuánto debe doler la quiebra no lo decidimos a ojo:
  es justo lo que estudia el experimento E4.
- **Idioma** del repo y del informe: inglés si lo queremos enseñar fuera de clase.

---

### 24 de noviembre: a cada cliente su tipo

Nos dimos cuenta de que un banco real no te cobra lo mismo a ti que a una empresa, ni a un hogar
muy endeudado que a uno que apenas debe. Pero tampoco podíamos dejar que la red eligiera un tipo
para cada uno de los 1050 clientes: no aprendería nada y, al pasar de 100 a 1000 hogares, la red
ya no serviría.

Así que el banco no decide los tipos uno a uno, **decide la regla**, como hace un banco de verdad.
Cada año fija cinco cosas: lo que paga por los depósitos, su margen para empresas, su margen para
hogares, cuánto más cobra a quien va más endeudado de lo normal, y hasta qué deuda acepta prestar
a un hogar. Con eso cada cliente acaba pagando su propio tipo, con un máximo de base + 15%, como
un límite de usura.

Al gobierno no le pone tipo: el banco compra su deuda al precio que marca el mercado, como pasa
en la realidad. Ahí no hay nada que aprender.

---

### 23 de noviembre: una economía que se parezca a la real

La pregunta que guía todo es: **¿puede un banco aprender con RL a quién prestar, cuánto y a qué
tipo, mejor que unas reglas fijas?** Para que la respuesta valga algo, la economía tenía que
parecerse a la de verdad.

- **Los hogares empiezan con su deuda real.** El EconGym original ignora la deuda de la encuesta
  (SCF 2022), así que nadie empezaba debiendo nada y el banco no tenía a quién prestar. En los
  datos reales, tres de cada cuatro familias tienen deuda.
- **1000 hogares y 50 empresas**, la proporción real de unos 20 hogares por empresa. El original
  tiene una sola empresa gigante, y así no hay forma de elegir a quién prestar. La partimos en 50
  empresas de tamaños distintos y con su propia suerte cada año.
- **Las reglas de los bancos de verdad.** Capital mínimo del 10,5% de lo prestado (Basilea III),
  y empiezan con un 15%, como los bancos europeos. Si el capital llega a cero, el banco quiebra.
- **No le damos pistas.** Todos los préstamos cuentan igual para el capital mínimo. En la vida real
  hay "pesos por riesgo", pero eso sería decirle al banco quién es peligroso, y eso es lo que tiene
  que aprender solo.
- **Lista de morosos.** Quien no paga entra en la lista de ese banco y ya no le presta. Es la
  información que tiene el banco para aprender. Si compartirla entre bancos ayuda o no es el
  experimento E7.
- **Solo cuentan los ahorros que están en el banco.** Al principio contábamos todo lo que tenían
  las familias (casas, acciones...) como depósitos, y el banco tenía tanto dinero que no podía
  quebrar nunca. Con solo el dinero en cuenta (~5% de sus activos) ya hay riesgo de verdad.
- **Episodios de 30 años**, porque la economía del original se viene abajo sola hacia el año 70.
- **El banco central sube y baja el tipo poco a poco** (regla de Taylor suavizada). Sin suavizar
  saltaba de 0% a 10% de un año a otro, algo que no pasa en la realidad.

---

### Antes: el planteamiento

- **Los bancos crean dinero al prestar**, como en la banca real. El límite no son las reservas sino
  el capital.
- **El más barato se lleva más clientes, pero no todos.** Si el más barato se lo llevara todo,
  sería una guerra de precios imposible de aprender.
- **Quebrar tiene que doler.** Si no, al banco le puede salir a cuenta arriesgarlo todo.
- **Un mismo PPO para uno o varios bancos**, para poder comparar. SB3 solo para comprobar que el
  entorno se puede aprender.
- **No tocamos el EconGym original:** lo nuestro está aparte y hereda de él, para que se vea claro
  qué hemos hecho nosotros.
