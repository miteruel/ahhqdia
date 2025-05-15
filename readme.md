

AHHQDIA

Bienvenidos al repositorio del equipo “Ahí hay hombres que dicen IA” (o AHIA por abreviar)  
Nuestro equipo AHHQDIA inicial  es :

* J \- BioTecnólogo, programador Junior con experiencia en IA  
* L \- Programador Senior de Sistemas, con experiencia en Microcontroladores, VideoCam, Arduino. Experiencia en IA y en varios lenguajes de programación  
* T-  Scrum Master   
* X \-  Dirección del proyecto. Que es un personaje que no conocemos y nos marca las pautas de tiempos y necesidades.

# FabuloZoo

   Van a montar un zoológico en nuestra ciudad, donde van a haber varias especies de  animales sueltos en varios sectores del parque. Se tiene unas cámaras captan imágenes de los sectores. También pueden haber sensores, alarmas y otros artilugios Arduino, pero eso hay que estudiarlo.

   Se necesita un programa que controle que animales y personas  hay en cada momento en cada sector, El programa  pueda dar diferentes alarmas.

Si hay demasiados animales en un sector, si hay especies exteriores que se hayan colado en el parque y sean peligrosas, etc..   
Pueden haber otras alarmas cuando una persona entra en un sector ‘peligroso’.  Pero aún se está discutiendo cómo hacerlo para no confundirlo con un trabajador de mantenimiento.  
Hay otros temas en estudio, pero se quiere comprobar la eficacia de la empresa contratada en el desarrollo del primer módulo. 

Como en la Diputación son muy pitos, nos han encargado el trabajo a nosotros, el equipo AHHQDIA. Que tenemos que desarrollar toda la parte del proyecto que podamos ☹️.  
Más que resultados se pide un estudio de viabilidad. Con documentación técnica de herramientas usadas 

Vamos a seguir una metodología Agile sencilla (sin volvernos locos, pero respetando lo principal),:

El director X  del proyecto es el encargado del primer paso:

**Crear una visión del producto**: Desarrolla una visión clara del producto o servicio que se entregará. Esto puede incluir un roadmap inicial y una lista de características prioritarias.

El objetivo es desarrollar un prototipo MVP lo más sencillo posible pero funcional.  
MVP:( Minimum Viable Product) [Producto viable mínimo \- Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/Producto_viable_m%C3%ADnimo)

* Los prototipos los desarrollamos en Python.  
* Necesariamente va a haber un  módulo de reconocimiento de imágenes de animales. Como prototipo, debería reconocer al menos 3 o 4 especies diferentes (+ personas?) .Como requerimiento inicial usaremos algoritmos YOLO [Algoritmo You Only Look Once (YOLO) \- Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/Algoritmo_You_Only_Look_Once_\(YOLO\))  
  Preferiblemente usando la librería open de Ultralytics  [Inicio \- Ultralytics YOLO Docs](https://docs.ultralytics.com/es/)  
* Hay que desarrollar también la parte “lógica funcional” que interprete los resultados de la IA de visión y produzca acciones.. Inicialmente sería programación tradicional, pero también se puede estudiar que haya una versión neuronal.  
* Hay que estudiar qué herramientas permiten conexión python con Arduino.  
* X irá ampliando información y detalles a medida que evolucione el proyecto.

El [Scrum Master](https://es.wikipedia.org/wiki/Scrum_\(desarrollo_de_software\)) ( es un rol dentro de un equipo [Agile](https://es.wikipedia.org/wiki/Desarrollo_%C3%A1gil_de_software)). Actúa como facilitador y líder del equipo Scrum (no es el jefe pero organiza el equipo). Algunas de sus responsabilidades incluyen:

* **Facilitar reuniones** como la planificación de sprints, las reuniones y las retrospectivas.  
* **Eliminar obstáculos** que puedan afectar el progreso del equipo.  
* **Promover la colaboración** entre los miembros del equipo y con los promotores.  
* **Entrenar al equipo** en los principios de Scrum y fomentar la autoorganización.  
* **Definen el propósito y los objetivos de los miembros**: Asegúrate de que todos los involucrados entiendan el propósito del proyecto y los resultados esperados.

**Todos:**  
**Realizar una reunión inicial**: Para conocer al equipo, establecer expectativas y resolver dudas. y **Formar el equipo ágil**: Equipo multifuncional con las habilidades necesarias. Define roles (Product Owner, Scrum Master y los miembros del equipo de desarrollo).  
En nuestro caso:  
J es el encargado del módulo de IA   
L es El programador principal.  
T actuará como Scrum Master,

**Planificar iteraciones**: Divide el trabajo en sprints o iteraciones cortas, generalmente de 1 a 4 semanas, y prioriza las tareas más importantes.

Sprint1  
J es el encargado del modulo de IA. Se documentará que partes de YoLo son necesarias para crear y entrenar un modelo. Buscará que fuentes de datos puede usar, documentará que formatos usan los entrenamientos. Buscará si hay modelos entrenados en HF que resulten interesantes  Inicialmente no es necesario que cree ninguna script o programa,pero debe preparar un documento jupyter, con los enlaces y descripciones , 

L prepara otro documento jupyter preparando una primera versión de interfaz de usuario con gradio (inicialmente para una sola camara), no tiene que estar conectada inicialmente a una IA real, puede ser un MOCK.  También se le solicita que se documente de que librerías o entornos serian necesarios para hacer una posible conexión entre Arduino y python, si eso es posible…

Tanto J como L pueden buscar en Kaggle o Github  si hay proyectos parecidos, y si los hay, adjuntar los enlaces en los documentos.

Sprint2

Se preparan los scripts de entrenamiento , las fuentes de datos..   
Se hacen pruebas con pocas iteraciones y con conjuntos pequeños de datos para probar los programas.  
Hay que analizar como se tienen que tomar los datos para dar las alarmas, como encajar la salida del modelo para que funcione como sensor.

En la parte de programación participa todo el equipo:

Se planifica como se hace el entrenamiento, dependiendo del tamaño de datos si es en la nube o en local.

Sprint3  
Se hace el entrenamiento del modelo. Se exporta a diferentes formatos de modelos.   
Se prueba la integración en la interface grafica

Sprint4  
Android con la version tfLite del modelo entrenado.

[Algoritmo You Only Look Once (YOLO) \- Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/Algoritmo_You_Only_Look_Once_\(YOLO\))

[Python GUI](https://pythongui.org/)

[AnimalCLEF25 @ CVPR-FGVC & LifeCLEF | Kaggle](https://www.kaggle.com/competitions/animal-clef-2025)

FabuloZoo

