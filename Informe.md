# Hand written recognition

## **Introducció**
El reconeixement d'escriptura a mà és una tasca complexa que implica la identificació i interpretació de caràcters escrits a mà. Aquesta tasca és fonamental en moltes aplicacions, com ara la digitalització de documents, la classificació automàtica de formularis i el reconeixement de signatures. <br/>
El reconeixement de caràcters normalment funciona bé en tipus de lletra impresos a màquina. No obstant això, encara suposa un repte difícil per a les màquines reconèixer caràcters escrits a mà, a causa de la gran variació dels estils d'escriptura individuals.
Les tècniques tradicionals basades en algorismes de processament d'imatges han estat reemplaçades gradualment per enfocaments basats en xarxes neuronals profundes, degut a la seva capacitat per aprendre característiques complexes directament des de les dades.
<br/><br/><br/>


## **Objectius**
L’objectiu consta al reconeixement de textes, concretament de noms propis escrits de forma manual. Això es duu a terme mitjançant un model de xarxes neuronals, que pugui reconèixer i interpretar text utilitzant tècniques avançades d'aprenentatge profund.
<br/><br/><br/>


## **Dataset**
El conjunt de dades utilitzat pel reconeixement d'escriptura a mà es pot obtenir a través del link: https://www.kaggle.com/datasets/landlord/handwriting-recognition/suggestions?status=pending&yourSuggestions=true. Aquest conjunt de dades consta de més de quatre-cents mil noms escrits a mà, recollits a través de projectes benèfics, amb l'objectiu de proporcionar una base robusta per a l'entrenament de models de reconeixement de text manuscrit. En total, hi ha 206.799 noms i 207.024 cognoms. <br/>
Respecte el contingut del conjunt de dades, les dades d'entrada consisteixen en centenars de milers d'imatges de noms escrits a mà, transcrites i dividides en conjunts de prova, formació i validació. Les imatges segueixen un format de nom que permet ampliar el conjunt de dades amb dades pròpies si es desitja. <br/>
L’estructura del conjunt de dades es basa en imatges i en arxius csv. Per un costat, les imatges corresponen a noms escrits a mà, distribuïts en tres subconjunts: d'entrenament (331.059), de proves (41.382) i de validació (41.382). D’altra banda, per tal d’avaluar el model, hi ha tres arxius CSV que contenen els noms correctes assignats a cada imatge de cada subconjunt específic: written_name_train_v2.csv per al conjunt d'entrenament, written_name_validation_v2.csv per al conjunt de validació i written_name_test_v2.csv per al conjunt de proves. <br/>
Cada arxiu csv té dues columnes: "FILENAME", que indica el nom de la imatge, i "IDENTITY", que representa el nom escrit a la imatge.
<br/><br/><br/>


## **Xarxes neuronals - MARC TEÒRIC**
Per abordar la tasca presentada prèviament, inicialment es va realitzar un plantejament envers les xarxes neuronals més aptes per al reconeixement de textos. Aquesta tasca implica la conversió d'imatges de text manuscrit en una representació textual que pugui ser processada per màquines. Per abarcar aquest problema, s'utilitzen arquitectures de xarxes neuronals convolucionals (CNN) i xarxes neuronals recurrents (RNN), combinades en una xarxa coneguda com a CRNN (Convolutional Recurrent Neural Network). <br/>
Cada tipus de xarxa aporta característiques complementàries que, combinades, proporcionen una solució potent i eficaç per a la tasca a realitzar. Per un costat, la CNN s’encarrega de l’extracció de les característiques, detectant patrons locals en les imatges, com vores, corbes i textures. En contrast, la RNN permet capturar dependències seqüencials a llarg termini, essent capaces d’adaptar el reconeixement de text en funció del context. Per exemple, si es prediu la paraula "HOIA", la RNN detectarà l'error i substituirà la "I" per la "L", predint la paraula "HOLA". Per tant, la RNN millora la precisió en predir la paraula. <br/>
En resum, la xarxa neuronal CRNN és un model híbrid entre CNN i RNN, que combina la capacitat de les CNN’s per extreure característiques visuals i de les RNN’s per modelar seqüències. <br/><br/>


Prèviament a realitzar la implementació (Vegeu Apartat Models - MARC PRÀCTIC), es duu a terme una explicació de les xarxes convolucionals (CNN), xarxes neuronals recurrents (RNN) i CRNN (xarxes neuronals recurrents convolucionals), per tal d’establir una base sobre que desenvolupar el marc de treball. <br/>

Les CNN funcionen imitant el sistema visual del cervell humà, amb capes que s'especialitzen a detectar característiques cada cop més complexes. Les primeres capes d'una CNN poden detectar línies i vores bàsiques, mentre que les capes superiors combinen aquestes característiques bàsiques per formar representacions més complexes, com ara rostres o vehicles. Aquest procés s'aconsegueix mitjançant l'ús de convolucions, on s'apliquen filtres (coneguts com a kernels) a les imatges d'entrada per extreure'n característiques rellevants. Els nuclis s'ajusten durant l'entrenament de la xarxa per maximitzar la capacitat de la xarxa per distingir entre diferents tipus d'imatges. <br/>
Conforme el processament avança a través de les capes, les característiques es tornen més complexes, i sorgeixen les capes convolucionals més profundes com les que se'n consideren les representacions més essencials dels atributs de les imatges. <br/>
Aquestes xarxes neuronals poden tenir desenes o centenars de capes, i cadascuna aprèn a detectar diferents característiques d'una imatge. S'apliquen filtres a les imatges d'entrenament amb diferents resolucions, i la sortida resultant de convolucionar cada imatge s'empra com a entrada per a la capa següent. Els filtres poden començar com a característiques molt simples, com ara brillantor i vores, i anar creixent en complexitat fins a convertir-se en característiques que defineixen l'objecte de forma singular. <br/>
Una CNN consta d’una capa d’entrada, una capa de sortida i diverses capes ocultes entre ambdues (Vegeu Fig. 1). <br/>
<img width="542" alt="Captura de pantalla 2024-06-04 a las 1 38 03" src="https://github.com/DCC-UAB/XNAPproject-grup_06/assets/91673341/eba7ef28-0b0d-4273-9985-09fc916b417c">
<br/> Fig 1. Arquitectura d’una xarxa CNN <br/>


Aquestes capes fan operacions que modifiquen les dades, amb el propòsit de comprendre'n les característiques particulars. Les 3 capes més comunes són: convolució, activació o ReLU, i agrupació. Per un costat, la capa de convolució aplica un conjunt de filtres convolucionals a les imatges d’entrada; cada filtre activa diferents característiques de les imatges. A continuació, la unitat lineal rectificada (ReLU), coneguda com a activació, manté els valors positius i estableix els valors negatius en zero, que permet un entrenament més ràpid i eficaç. Finalment, la capa d’agrupació, conegudes com a capes de pooling, redueixen la dimensionalitat i s’encarreguen de descartar informació espacial i detalls irellevants. <br/> 
Aquestes operacions es repeteixen en desenes o centenars de capes; cada capa aprèn a identificar diferents característiques (Vegeu Fig. 2). <br/>
<img width="685" alt="Captura de pantalla 2024-06-04 a las 1 39 55" src="https://github.com/DCC-UAB/XNAPproject-grup_06/assets/91673341/b0d4cc37-244a-493e-a972-d683a7bf4758">
 <br/> Fig 2. Exemple de xarxa amb múltiples capes convolucionals <br/> <br/>


En contraposició, una xarxa neuronal recurrent (RNN) és una estructura d'aprenentatge profund que utilitza informació passada per millorar el rendiment de la xarxa en les entrades actuals i futures. El que distingeix un RNN és la seva capacitat per mantenir un estat intern i utilitzar bucles ocults. Aquesta estructura (Vegeu Fig. 3) de bucle permet a la xarxa emmagatzemar informació passada en estat ocult i operar en seqüències. <br/>
<img width="417" alt="Captura de pantalla 2024-06-04 a las 1 41 14" src="https://github.com/DCC-UAB/XNAPproject-grup_06/assets/91673341/5f59f701-0656-4f3b-941f-b92ac2aa779a">
<br/> Fig 3. Arquitectura d’una xarxa RNN <br/>

La xarxa RNN aplica la informació passada a l'entrada actual mitjançant l'ús de dos conjunts de pesos: un per al vector d'estat ocult i un altre per a les entrades. Durant l'entrenament, la xarxa aprèn a ajustar aquests pesos per a les entrades i l'estat ocult. En la implementació, la sortida es genera considerant tant l'entrada actual com l'estat ocult, que a la vegada es basa en les entrades anteriors. <br/>
A la pràctica, els RNN simples sovint es troben amb el problema de l'aprenentatge de dependències a llarg termini. En el seu entrenament, s'utilitza habitualment la retropropagació, però s'hi pot observar un fenomen de gradient "desapareixent" o "explotant". Aquestes dificultats poden resultar en pesos de la xarxa que es tornen molt petits o molt grans, limitant la capacitat del RNN per aprendre relacions a llarg termini de manera efectiva. Per superar aquesta limitació, s'ha desenvolupat un tipus especial de RNN conegut com a xarxa de memòria a llarg termini (LSTM). Les xarxes LSTM incorporen portes addicionals que regulen el flux d'informació entre l'estat ocult i la sortida, així com cap a l'estat ocult següent. Aquest mecanisme permet a la xarxa aprendre relacions a llarg termini amb més eficàcia en les dades. <br/><br/>


En última instància, és necessari mencionar la CRNN, que consta el nucli d’aquest projecte. Les xarxes neuronals recurrents convolucionals o CRNN, s'utilitzen normalment per processar i classificar dades de seqüències com ara veu, text i imatges. La seva capacitat per manejar dades seqüencials de longitud variable i capturar dependències a llarg termini els fa especialment efectius en tasques que requereixen comprendre i modelar informació contextual i temporal, com ara el reconeixement de textos. <br/>
El funcionament dels CRNN es basa en el processament d'una seqüència d'entrada, que pot ser imatges o mostres d'àudio. Aquesta seqüència es passa a través de capes convolucionals, similars a les utilitzades en les CNN, que són especialment efectives per a les entrades basades en imatges. Després, la sortida d'aquestes capes convolucionals alimenta una o més capes recurrents, que destaquen per la seva eficàcia en el tractament de dades seqüencials. Les capes recurrents mantenen un estat ocult que captura la informació de les entrades anteriors de la seqüència. Les connexions entre les capes convolucionals i recurrents són crucials; sovint, la sortida d'una capa convolucional es mostreja abans de ser introduïda a una capa recurrent, reduint la complexitat computacional de la xarxa i conservant les característiques essencials de l'entrada. Finalment, la sortida de l'última capa iterativa passa a través d'una capa final completament connectada, que produeix una predicció per a la seqüència d'entrada. Aquesta predicció pot comprendre una seqüència de caràcters, paraules o altres sortides rellevants per a la tasca. <br/><br/>


Addicionalment, es va plantejar la possibilitat d’implementar altres models per tal de dur a terme aquesta tasca, com ara MLP, YOLO, POCHNET, Transformer o Metric Learning. Per un costat, en aquest context, la MLP suposa una xarxa excessivament simple, donat que manquen de la capacitat per processar dades seqüencials i incapacitat per capturar relacions espacials en imatges. No obstant això, s’ha considerat que, tant el sistema de detecció d'objectes YOLO com la  xarxa neuronal convolucional CNN PHOCNet, representen opcions molt complexes per implementar. En últim lloc, s’ha descartat l’ús de Transformer o Metric Learning, donat que s’ha prioritzat la implementació de la xarxa CRNN. 
<br/> <br/> <br/>


## **Models - MARC PRÀCTIC**
Després d'una anàlisi exhaustiva del nostre conjunt de dades, s’ha decidit no aplicar tècniques de data augmentation en el procés d'entrenament del nostre model de reconeixement de noms escrits a mà. La nostra decisió es basa en les següents raons:<br/>
- El conjunt de dades d'entrenament utilitzat consta de 331.059 imatges, un volum que considerem prou gran per garantir una representació adequada de la variabilitat inherent en l'escriptura a mà. A més, els conjunts de proves i de validació també són amplis, amb 41.382 imatges cadascun, proporcionant així una base sòlida per a l'avaluació del model.<br/>
- Una inspecció detallada del conjunt de dades ha revelat que les imatges inclouen una àmplia gamma d'estils d'escriptura, il·luminacions i altres factors de variació. Aquesta diversitat ja existent dins el conjunt de dades d'entrenament és suficient per a què el model pugui aprendre a generalitzar adequadament sense necessitat d'augmentar artificialment les dades.<br/>
- L'augmentació de dades implica un processament addicional que pot augmentar el temps d'entrenament i el cost computacional. Atès que el conjunt de dades utilitzat és ja considerablement gran, s’ha optat per evitar aquest cost addicional i concentrar els recursos disponibles en l'optimització d'altres aspectes del model.<br/>
- Les proves preliminars amb el conjunt de dades existent han mostrat que el model aconsegueix un rendiment sòlid amb una alta precisió en el reconeixement dels noms escrits a mà. La capacitat del model per generalitzar a noves dades sembla adequada segons les mètriques de validació, la qual cosa suggereix que el conjunt de dades actual és suficient per a les nostres necessitats.
<br/><br/>

(CNN imatges per lletres segmentades)<br/>
En el reconeixement de noms escrits a mà amb imatges de mida 14x10, el disseny d'una xarxa neuronal convolucional (CNN) amb poques capes pot ser suficient per aconseguir un bon rendiment degut a la mida petita de les imatges i la simplicitat de la tasca.<br/>
Amb una resolució de només 14x10 píxels, les imatges tenen una quantitat limitada d'informació. Això implica que no es requereixen moltes capes per extreure les característiques rellevants, ja que les convolucions i les capes de pooling poden capturar tota la informació útil en poques etapes.<br/>
El reconeixement de noms escrits a mà és menys complex que altres tasques. Les característiques necessàries per distingir entre diferents caràcters o paraules poden ser capturades amb menys capes convolucionals.<br/>
Utilitzar moltes capes en un model amb un conjunt de dades limitat pot conduir a un sobreajustament, on el model aprèn patrons específics de les dades d'entrenament però no generalitza bé a noves dades. Un model més simple amb poques capes és menys propens a aquest problema.<br/>
Les capes essencials per el reconeixement de noms escrits a mà són les següents:<br/>
- Capes convolucionals (Conv2D): Extreuen característiques locas de les imatges (vores, textures…). Una capa convolucional és necessària per detectar les característiques bàsiques de les lletres.<br/>
- Capes de pooling (MaxPooling2D): Redueixen la dimensió espacial de les característiques extretes, reduint la complexitat computacional i afegint informació. Una capa de pooling després de la capa convolucional ajuda a reduir la mida de les característiques i a fer el model més robust a petites translacions.<br/>
- Capes completament connectades (Fully Connected): Combina les característiques extretes per fer la classificació final. Una capa completament connectada és necessària per combinar les característiques i produir la sortida final.<br/>
- Capes de sortida (Output Layer): Genera les possibilitats de les diferents classes (lletres o paraules). Una capa de sortida és necessària amb el nombre adequat de neurones per al nombre de classes. 

<br/><br/><br/>


## **Avaluació dels models**
Per tal d’avaluar els models es va plantejar l’ús de diferents mètriques, com ara: Accuracy o  Word Error Rate (WER), Precision, Recall, F1-score, l'error de reconeixement de caràcters (Character Recognition Error). En aquest cas, Accuracy i Word Error Rate (WER) representen el mateix paràmetre donat que mesuren la precisió del model en la predicció dels noms escrits a mà. <br/>
Així doncs, l’Accuracy, o Word Error Rate (WER), mesura el nombre de prediccions correctes realitzades respecte a la proporció total de prediccions fetes, oferint una visió general de com aprèn el model. Per un costat, l’indicador Precision quantifica el nombre de prediccions positives realitzades correctament respecte al nombre total de casos positius, permetent avaluar la capacitat del model per identificar correctament els casos positius. D’altra banda, Recall representa la proporció de tots els casos positius reals que el model ha identificat correctament, mesurant la seva capacitat per identificar correctament els casos positius reals. L’indicador F1-score combina les dues mètriques anteriors, Precision i Recall, proporcionant una mesura global de la capacitat del model per identificar correctament les instàncies positives i evitar falsos positius. Finalment, Character Recognition Error avalua  l'error de reconeixement de caràcters. <br/>
Addicionalment, es va considerar la creació d’un HeatMap per visualitzar i analitzar les regions de les imatges on el model té més o menys precisió. <br/><br/>

En haver dut a terme una recerca sobre les mètriques anteriors per avaluar els models de reconeixement d'escriptura a mà, s'ha optat per utilitzar l'Accuracy i l'Error de Reconeixement de Caràcters. <br/>
En contrast, es va optar per no fer servir els indicadors Precision, Recall i F1-score.  Això és degut que, donada la quantitat de classes amb què es treballen (concretament 29), aquestes mesures són excessivament detallades. En contraposició, es va decidir utilitzar HeatMap, en comptes de les mètriques anteriors. Això és degut que ofereix una alternativa visualment potent i específica per avaluar el rendiment del model. <br/><br/>


D'altra banda, a l'hora d'avaluar l'aprenentatge del model, es té en compte el subajustament (underfitting) i el sobreajustament (overfitting). <br/>

Aquests dos conceptes es tenen en compte mitjançant les gràfiques de "loss". Aquestes, són eines essencials per avaluar el rendiment i la salut del model de reconeixement de noms escrits a mà al llarg del temps, ajudant a prendre decisions informades per millorar-lo. Aquestes gràfiques permeten veure com progressa l'entrenament del model, la qual cosa ajuda a determinar si el model està aprenent de manera adequada o si està tenint problemes. <br/>
Un dels problemes amb els quals es podria trobar el model és el sobreajustament. Aquest fenomen es produeix quan el model aprèn massa bé els detalls i el soroll de les dades d'entrenament, a costa del seu rendiment en dades noves (de test). Si la "loss" d'entrenament continua disminuint mentre la "loss" de test comença a augmentar, és una senyal clara de sobreajustament. Les gràfiques de "loss" ajuden a detectar aquest tipus de problemes i a prendre mesures, com ara utilitzar tècniques de regularització o aturar l'entrenament abans.<br/>
D'altra banda, el subajustament és un altre dels problemes amb els quals es podria trobar el model. Aquest fenomen es produeix quan el model és massa senzill per capturar les relacions subjacents en les dades. Si tant la "loss" d'entrenament com la de test es mantenen altes i no disminueixen suficientment, és una senyal de subajustament. Això pot indicar la necessitat d'un model més complex.<br/>
A més, les gràfiques de "loss" són útils per ajustar els hiperparàmetres del model, com ara la mida del lot (batch size), la taxa d'aprenentatge (learning rate) i el nombre d'èpoques d'entrenament. Es pot veure l'impacte dels canvis en aquests paràmetres directament en les gràfiques. Aquesta informació proporcionada per les gràfiques pot ser de gran ajuda per optimitzar i perfeccionar l’entrenament del model utilitzat.



<br/><br/><br/>


## **Selecció del model**
<br/><br/><br/>


## **Problemes i solucions**
<br/><br/><br/>


## **Conclusió**
<br/><br/><br/>


## **Bibliografia**
[1] Harald Scheidl, S. Fiel, and R. Sablatnig, “Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm,” Aug. 2018, doi: https://doi.org/10.1109/icfhr-2018.2018.00052. <br/>
[2] Diplom-Ingenieur, H. Scheidl, and R. Sablatnig, “Handwritten Text Recognition in Historical Documents DIPLOMARBEIT zur Erlangung des akademischen Grades Visual Computing eingereicht von.” Available: https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742 <br/>
[3] H. Scheidl, “Build a Handwritten Text Recognition System using TensorFlow,” Medium, May 22, 2023. https://towardsdatascience.com/2326a3487cd5 <br/>
[4] L. CARES, “Handwritten Digit Recognition using Convolutional Neural Network (CNN) with Tensorflow,” Medium, Aug. 03, 2022. https://learner-cares.medium.com/handwritten-digit-recognition-using-convolutional-neural-network-cnn-with-tensorflow-2f444e6c4c31 <br/>
[5] anandhkishan, “Handwritten-Character-Recognition-using-CNN/emnist cnn model.ipynb at master · anandhkishan/Handwritten-Character-Recognition-using-CNN,” GitHub, 2018. https://github.com/anandhkishan/Handwritten-Character-Recognition-using-CNN/blob/master/emnist%20cnn%20model.ipynb <br/>
[6] S. Gautam, “How to Make Real-Time Handwritten Text Recognition With Augmentation and Deep Learning,” Medium, May 16, 2023. https://sushantgautm.medium.com/how-to-make-real-time-handwritten-text-recognition-with-augmentation-and-deep-learning-9281323d80c1 <br/>
[7] Nightmare, “NightmareNight-em/Handwritten-Digit-Recognition,” GitHub, Dec. 21, 2019. https://github.com/NightmareNight-em/Handwritten-Digit-Recognition/tree/master <br/>
[8] Đ. Q. Tuấn, “tuandoan998/Handwritten-Text-Recognition,” GitHub, May 11, 2024. https://github.com/tuandoan998/Handwritten-Text-Recognition/tree/master <br/>
[9] “OCR Handwriting Recognition CNN,” kaggle.com. https://www.kaggle.com/code/ademhph/ocr-handwriting-recognition-cnn <br/>
[10] “Handwriting Recognition Using CNN Model,” kaggle.com. https://www.kaggle.com/code/ademhph/handwriting-recognition-using-cnn-model <br/>
[11] “Handwritten text Recognition using CNN,” kaggle.com. https://www.kaggle.com/code/pandeyharsh407/handwritten-text-recognition-using-cnn <br/>
[12] “Handwriting_Recognition_with_CRNN_Model,” kaggle.com. https://www.kaggle.com/code/quydau/handwriting-recognition-with-crnn-model <br/>
[13] P. Hoang, “huyhoang17/CRNN_CTC_English_Handwriting_Recognition,” GitHub, Feb. 21, 2023. https://github.com/huyhoang17/CRNN_CTC_English_Handwriting_Recognition/tree/master <br/>
[14] “Handwriting Recognition using CRNN in Keras,” kaggle.com. https://www.kaggle.com/code/ahmadaneeq/handwriting-recognition-using-crnn-in-keras  <br/>
[15] “Guardar y cargar modelos | TensorFlow Core,” TensorFlow. https://www.tensorflow.org/tutorials/keras/save_and_load?hl=es-419 <br/>
[16] “Saving and Loading Models — PyTorch Tutorials 1.4.0 documentation,” Pytorch.org, 2017. https://pytorch.org/tutorials/beginner/saving_loading_models.html <br/>
[17] “How to Predict Using a PyTorch Model | Saturn Cloud Blog,” saturncloud.io, Jul. 10, 2023. https://saturncloud.io/blog/how-to-predict-using-a-pytorch-model/#:~:text=To%20predict%20outcomes%20using%20the <br/>
[18]“¿Qué son las redes neuronales convolucionales? | 3 cosas que debe saber,” es.mathworks.com. https://es.mathworks.com/discovery/convolutional-neural-network.html#:~:text=Una%20CNN%20consta%20de%20una <br/>
[19] “Redes neuronales de memoria de corto-largo plazo - MATLAB & Simulink - MathWorks España,” es.mathworks.com. https://es.mathworks.com/help/deeplearning/ug/long-short-term-memory-networks.html <br/>
[20] “Convolutional Recurrent Neural Network For Text Recognition,” www.xenonstack.com. https://www.xenonstack.com/insights/crnn-for-text-recognition <br/>
[21] “What Is a Recurrent Neural Network (RNN)?,” es.mathworks.com. https://es.mathworks.com/discovery/rnn.html

<br/><br/><br/>


