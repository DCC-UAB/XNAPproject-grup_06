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
- CNN
- Divisió imatges i cració nou dataset
- CNN + RNN
- Plantejament de l'ús de xarxes com: YOLO, POCHNET, Transformer Metric learning, decartat.
- CRNN
<br/><br/><br/>


## **Models - MARC PRÀCTIC**
<br/>
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
[16] “Saving and Loading Models — PyTorch Tutorials 1.4.0 documentation,” Pytorch.org, 2017. https://pytorch.org/tutorials/beginner/saving_loading_models.html
<br/><br/><br/>


