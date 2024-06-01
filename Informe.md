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
<br/><br/><br/>


## **Avaluació dels models**
Per tal d’avaluar els models es va plantejar l’ús de diferents mètriques, com ara: Accuracy o  Word Error Rate (WER), Precision, Recall, F1-score, l'error de reconeixement de caràcters (Character Recognition Error). En aquest cas, Accuracy i Word Error Rate (WER) representen el mateix paràmetre donat que mesuren la precisió del model en la predicció dels noms escrits a mà. <br/>
Així doncs, l’Accuracy, o Word Error Rate (WER), mesura el nombre de prediccions correctes realitzades respecte a la proporció total de prediccions fetes, oferint una visió general de com aprèn el model. Per un costat, l’indicador Precision quantifica el nombre de prediccions positives realitzades correctament respecte al nombre total de casos positius, permetent avaluar la capacitat del model per identificar correctament els casos positius. D’altra banda, Recall representa la proporció de tots els casos positius reals que el model ha identificat correctament, mesurant la seva capacitat per identificar correctament els casos positius reals. L’indicador F1-score combina les dues mètriques anteriors, Precision i Recall, proporcionant una mesura global de la capacitat del model per identificar correctament les instàncies positives i evitar falsos positius. Finalment, Character Recognition Error avalua  l'error de reconeixement de caràcters. <br/>
Addicionalment, es va considerar la creació d’un HeatMap per visualitzar i analitzar les regions de les imatges on el model té més o menys precisió. <br/><br/>

En haver dut a terme una recerca sobre les mètriques anteriors per avaluar els models de reconeixement d'escriptura a mà, s'ha optat per utilitzar l'Accuracy i l'Error de Reconeixement de Caràcters. <br/>
En contrast, es va optar per no fer servir els indicadors Precision, Recall i F1-score.  Això és degut que, donada la quantitat de classes amb què es treballen (concretament 29), aquestes mesures són excessivament detallades. En contraposició, es va decidir utilitzar HeatMap, en comptes de les mètriques anteriors. Això és degut que ofereix una alternativa visualment potent i específica per avaluar el rendiment del model. <br/><br/>


D'altra banda, a l'hora d'avaluar l'aprenentatge del model, es té en compte el subajustament (underfitting) i el sobreajustament (overfitting). 

<br/><br/><br/>


## **Selecció del model**
<br/><br/><br/>


## **Problemes i solucions**
<br/><br/><br/>


## **Conclusió**
<br/><br/><br/>


## **Bibliografia**
<br/><br/><br/>


