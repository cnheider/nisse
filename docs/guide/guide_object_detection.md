# Introduktion
Denne guide beskrive hvordan du lettest kan komme igang med Object Detection i Tensorflow. Det forudsættes at du er nogenlunde fortrolig med Bash og Python, og har en egnet computer (se afsnittet nedenfor). Man kan argumentere for at Google's Tensorflow og Object Detection API udgør det mest modne framework idag, og danner derfor udgangspunkt for denne guide. Når du har været disse trin igennem, har du et fuldt object detection system som er trænet til at genkende dine egne objekter på dine egne billeder. Guiden er delt op i de fem logiske skridt man skal igennem for at implementere et fuldt system: 1) Først sørger vi for at computeren er kompatibel med Tensorflow, dvs. at den har de rette pakker og drivere. 2) Så indsamler, organiserer og annoterer vi dine data. 3) Vi forbereder en standard-model som allerede er for-trænet. Dette kaldes Transfer Learning, og kan spare rigtig meget træningstid. 4) Vi er nu klar til at træne det neurale netværk. 5) Til slut kan vi evaluere netværket og se hvor godt det genkender og lokaliserer objekter.

## Hvad er object detection?
Med *object detection* forstås et system, som er i stand til både at identificere og lokalisere objekter i billeder. Dette er således mere avanceret end *object classification*, som blot beskæftiger sig med at kunne sige hvad et billede forestiller, når der i udgangspunktet kun er ét objekt på hvert billede. Med object detection kan der være arbitrært mange objekter i hvert billede, og efter træning kan systemet både genkende hvor objekterne er, og hvad de forestiller. Se f.eks. denne illustration: 

![Eksempel på object detection](img/kites_detections_output.jpg "Eksempel på object detection")

# Opsætning af computer
Deep learning er kræver forholdsvis meget af hardwaren, og træning på store datasæt kan tage lang tid, selv på kraftige maskiner. Årsagen til at det er overhovedet er blevet gangbart at udføre deep learning på andet end super-computere er, at man har lært at udnytte hardware-acceleration fra grafikkort, som primært er udviklet til computerspil. Denne guide antager at din maskine har et CUDA-understøttet grafikkort (f.eks. fra Nvidia), da du ellers kun kan træne på CPU, hvilket er meget langsommere. Udover et godt grafikkort, er det også tilrådeligt at have en stor harddisk, gerne på et par terabyte. Til gengæld behøver den ikke være SSD. Hvis du ikke i forvejen har hardware der egner sig til deep learning, er der også gode muligheder for at køre beregningerne "i skyen". [Kontakt](#kontakt) evt. Alexandra Instituttet for at høre nærmere om disse muligheder.

Der er en udfordring i at opsætte kompatible versioner af den nødvendige software. Specifikt handler det om at have egnede versioner af styresystem/kernel, CUDA/CuDNN, og grafikkort-driver, som Tensorflow understøtter. Denne guide tager udgangspunkt i følgende versioner, som vi har bekræftet fungerer sammen:
- Styresystem: Ubuntu 16.04 (kernel: 4.15.0-36-generic) 
- CUDA: CUDA 9.0 (v9.0.176)
- CuDNN: 7.2.1
- GPU-driver: Nvidia 396.54
- Tensorflow: 1.11.0

Hvis du kører Ubuntu, kan du tjekke dine versioner med:
- Styresystem: `lsb_release -a`
- Kernel: `uname-r`
- CUDA: `nvcc --version`
- CuDNN: `cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2`
- Tensorflow: `pip list | grep tensorflow`

Det samlede system installeres i seks trin, nemlig Ubuntu, CUDA, CuDNN, GPU-driver, Tensorflow, Protobuf

## Installation af Ubuntu 16
TBD

## Installation af CUDA 9
TBD

## Installation af CuDNN
TBD

## Installation af GPU-driver
TBD

## Installation af Tensorflow

For at installere Tensorflow og andre nødvendige pakker, køres
`pip install --user tensorflow-gpu matplotlib Cython contextlib2 jupyter pillow lxml`

Nu hvor computeren har de nødvendige undersystemer, mangler vi blot at klone Tensorflow Models-kodebasen og installere et par Python-pakker. Det er heldigvis ganske smertefrit:
```bash
mkdir ~/Documents/tensorflow
cd ~/Documents/tensorflow
git clone https://github.com/tensorflow/models.git
```

Vi kører nu installationsprogrammet
```bash
cd ~/Documents/tensorflow/models/research/
python3 setup.py build
sudo python3 setup.py install
```

Det er nødvendigt at Models-kodebasen kan findes på Python-stien, hvilket gøres ved at skrive følgende i en terminal
`export PYTHONPATH=$PYTHONPATH:~/Documents/tensorflow/models/research:~/Documents/tensorflow/models/research/slim`

Dette skal gøres efter hver genstart af computeren, så alternativt kan ovenstående linien tilføjes til sidst i filen `~/.bashrc`, som kan redigeres med `sudo nano ~/.bashrc`

Det kan virke forvirrende, at vi både henter noget med Tensorflow via `pip` og via `git clone`. Når vi installerer `tensorflow-gpu` via `pip`, betyder det, at man nu kan kalde 
`import tensorflow as tf` i Python, og derefter have adgang til det fulde Tensorflow framework. Den Models-kodebase vi henter via `git clone` er som sådan ikke Tensorflow selv, men en masse kode der bygger ovenpå Tensorflow, og som indeholder en masse eksempler hvordan man kan bruge Tensorflow. 

## Kompilation af protobuf
For at få den rette version af protobuf gør vi sådan:
```bash
cd ~/Documents/tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
```

## Test installationen
Vi kan nu teste tensorflow-installation:
`python3 -c "import tensorflow as tf; print(tf.__version__)"`

Hvis der bliver outputtet et versionsnummer, er det bagvedliggende Tensorflow-system klar til at brug. Vi kan desuden teste at Models-kodebasen fungerer:
```bash
cd ~/Documents/tensorflow/models/research/
python3 object_detection/builders/model_builder_test.py
```

Du kan også køre den medfølgende object_detection_tutorial.ipynb for at se Tensorflow udføre inference på et test-billede
`jupyter notebook`. Vælg object_detection/object_detection_tutorial.ipynb og tryk Run

# Annotér dine data
Data-annotering ofte er den mest tidskrævende (og dermed dyre) del af processen, idet deep learning fungerer bedst med mange tusind billeder som alle skal annoteres. Alexandra Instituttets Visual Computing Lab er specialister i at syntetisere annotationer, så man slipper for manuel annotering (se afsnittet [Kontakt](#kontakt)). Selve annoteringen er udenfor denne guides afgrænsning, men et anbefalelsesværdigt værktøj er [LabelImg](https://github.com/tzutalin/labelImg).

Deep learning er typisk synonymt med enorme datamængder, og Tensorflow opererer allerhurtigst når data er organiseret i såkaldte Tensorflow-records, som er Googles eget format til at opbevare data. Der er også andre måder at organisere informationen, f.eks. i .csv- eller .xml-filer, som ofte efterfølgende konverteres til TF-record format. Det er særligt nyttigt at organisere dataene i .record-filer, hvis dataene ligger på en roterende harddisk (fremfor en SSD), da læsehastigheden bliver meget højere. Generelt er filer på adskillige gigabytes dog besværlige at håndtere, og Tensorflow har faktisk bedre mulighed for at parallelisere og randomisere data når de bliver delt op i flere .record-filer. Dette kaldes sharding, og håndteres automatisk i demonstrations-scriptet.

## Omdannelse af data til .record-format
TBD

Hvis du endnu ikke har egne data klar, eller blot ønsker at se hvordan systemet fungerer, inden du forsøger at træne på dine egne data, kan du følge nedenstående guide:

## Test på billeder af kæledyr

Download prøve-data
```bash
cd ~/Documents/tensorflow/models/research/
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
```

Der ligger nu to mapper, images og annotations, som hver især indeholder billeder af kæledyr og informationer om hvor på billedet et kæledyr er, og hvilken race det har. Tensorflow arbejder mest effektivt når data struktureres i såkaldte .record-filer. Vi kan skabe disse filer på følgende måde:
```bash
cd ~/Documents/tensorflow/models/research/
python3 object_detection/dataset_tools/create_pet_tf_record.py --label_map_path=object_detection/data/pet_label_map.pbtxt --data_dir=`pwd` --output_dir=object_detection/data
```

Bekræft at der nu ligger .record-filer i mappen `object_detection/data`.

Vi downloader nu en model, som allerede er blevet trænet på store mængder meget forskelligartede data, og som vi efterfølgende gen-træner på vores eget datasæt.
```bash
cd ~/Documents/tensorflow/models/research/object_detection/data
wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
mv faster_rcnn_resnet101_coco_11_06_2017/* .
rm -r faster_rcnn_resnet101_coco_11_06_2017/
```

Endelig skal Tensorflow have en konfigurations-fil, som fortæller hvor dataene ligger, hvilket neuralt netværk der skal trænes på, samt en masse parametre. Vi bruger her en konfigurations-fil som i forvejen egner sig til kæledyrs-datasættet, og vi skal blot indsætte den korrekte stil:
```bash
cd ~/Documents/tensorflow/models/research/
cp object_detection/samples/configs/faster_rcnn_resnet101_pets.config object_detection/data/faster_rcnn_resnet101_pets.config
sed -i "s|PATH_TO_BE_CONFIGURED|object_detection/data|g" object_detection/data/faster_rcnn_resnet101_pets.config
```

Bekræft at filen `object_detection/data/faster_rcnn_resnet101_pets.config` har følgende sti under `fine_tune_checkpoint`: `"object_detection/faster_rcnn_resnet101_coco_11_06_2017/data/model.ckpt"`

Du kan nu springe videre til afsnittet [Træning](#træning)

# Hent en model
Der findes en oversigt over tilgængelige, færdig-trænede modeller på i Tensorflow's [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

I denne guide bruger vi SSD Mobilenet v1 trænet på [COCO-datasættet](http://cocodataset.org):
```bash
cd ~/Documents/tensorflow/models/research/object_detection
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

Ekstraher i object_detection-mappen via 
`tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz`

# Opsæt konfiguration
Opret my_labels.pbtxt i object_detection/data/ med formatet:
```bash
item {
  id: 1
  name: 'annotationens_navn'
}
```
hvor id er et unikt tal startende fra 1, som bruges til at oversætte mellem labels (tal) og tekstbeskrivelse. Et eksempel kunne være:

```bash
item {
  id: 1
  name: 'hund'
}
item {
  id: 2
  name: 'kat'
}
```

Kopiér en konfiguration:
```bash
cd ~/Documents/tensorflow/models/research/object_detection
cp samples/configs/ssd_mobilenet_v1_coco.config data/my_training_config.config
```

Åbn konfigurationsfilen i en tekseditor, og foretag følgende justeringer:
- indstil hvor mange klasser du har, dvs. hvilket tal der er dit højeste id-label (i ovenstående eksempel er "kat" den sidste label, med tallet 2). Under `model, ssd` udfyldes f.eks. `num_classes: 2`
- Vi skal nu indstille batch-size, som er antallet af billeder der behandles i hver iteration. Hvis tallet bliver for stort, løber man tør for hukommelse på grafikkortet. Hvor stort tallet kan være afhænger således både af sit grafikkort, og af hvor store billederne er. Under `train_config` skrives f.eks. `batch_size: 8`
- I afsnittet `root` indtaster du den relative sti (fra research-mappen) til det model-checkpoint der skal trænes fra. Hvis du har gjort som beskrevet ovenfor, kan du skrive: `fine_tune_checkpoint: "object_detection/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"`
- Vi skal også oplyse stien til de .record-filer, der skal bruges til træning og evaluering. Bemærk, at det sidste tal i stien (f.eks. 00008 og 00002 i eksemplet) skal svare til det totale antal shards. Under `train_input_reader, tf_record_input_reader` skrives f.eks. `input_path: "data/train.record-?????-of-00008"` og under `eval_input_reader, tf_record_input_reader` skrives f.eks. `input_path: "data/test.record-?????-of-00002"`. Bevar `?????`, som instruerer Tensorflow i at denne del af filnavnet varierer. 
- Endelig, skal `label_map_path` angives to steder, for hhv. træningsdata og evalueringsdata. Her bruger vi stien til det label-map der blev fremstillet ovenfor, så f.eks. `label_map_path: "data/my_labels.pbtxt"` begge steder.
  
# Træning
For at træne (køres fra research-mappen)
```bash
cd ~/Documents/tensorflow/models/research/
python3 object_detection/model_main.py --pipeline_config_path=object_detection/data/my_training_config.config --model_dir=object_detection/models/model
```

*Bemærk:* Da der blev oplevet problemer med at Tensorflow går i stå hver gang den har gemt et checkpoint, blev model_main.py modificeret med et save_every_n_hours input argument, så man nemmere kan vælge hvor ofte der skal gemmes checkpoints. I så fald bliver kaldet:
`python3 object_detection/model_main.py --pipeline_config_path=object_detection/data/my_training_config.config --model_dir=object_detection/models/model --save_every_n_hours=8`. **TODO:** Upload modificeret model_main.py

Man kan følge med i træningen via Tensorboard:
`tensorboard --logdir=/home/ktolbol/Documents/tensorflow2/models/research/object_detection/models/model`

# Inference
Når træningen er stoppet, kan man fastfryse vægtene og bruge dem til "inference" på nye billeder. Bemærk, at `TRAINED_CKPT_PREFIX` angiver filnavnet på det sidste checkpoint, som afhænger af hvor længe modelle blev trænet. Erstat spørgsmålstegnet med det relevante tal, f.eks. `model.ckpt-123456` :
```bash
cd ~/Documents/tensorflow/models/research
mkdir object_detection/models/frozen_model
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=object_detection/data/my_training_config.config
TRAINED_CKPT_PREFIX=object_detection/models/model/model.ckpt-?
EXPORT_DIR=object_detection/models/frozen_model
python3 object_detection/export_inference_graph.py \
  --input_type=${INPUT_TYPE} \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
  --output_directory=${EXPORT_DIR}
```

Den fastfrosne model ligger nu i `object_detection/models/frozen_model`, og er klar til at blive brugt til inference. Vi kan lægge et par test-billeder i en mappe:
`mkdir object_detection/my_test_images`

Læg i denne mappe to billeder som du ønsker at teste på, navngivet som `image1.jpg` og `image2.jpg`.

Vi kan nu afprøve inference ved at køre Object Detection API'ets egen tutorial notebook `object_detection/object_detection_tutorial.ipynb`, på de to nye billeder:

i `object_dection_tutorial.ipynb`, ændres:
- `PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'` til `PATH_TO_FROZEN_GRAPH = 'models/frozen_model/frozen_inference_graph.pb'`
- tilsvarende ændres `PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')` til `PATH_TO_LABELS = 'data/my_labels.pbtxt'`
- endelig ændres `PATH_TO_TEST_IMAGES_DIR = 'test_images'` til `PATH_TO_TEST_IMAGES_DIR = 'my_test_images'`

Undlad at køre "Download model" segmentet i notebook'en, men ellers kør den igennem og se hvordan systemet klarer sig.

Forhåbentlig er det nu lykkedes dig at køre Google's Object Detection på dine egne data, men Tensorflow kan bruges til utroligt meget andet også, og der er mange andre netværksarkitekturer med hver deres fordele og ulemper, som man kan benytte - eller man kan opbygge et skræddersyet netværk. Alexandra Instituttet hjælper gerne med den videre udvikling (se [Kontakt](#kontakt)).

# Kontakt
Ved spørgsmål og henvendelser angående muligt samarbejde, kontakt Kristian Tølbøl: kristian.tolbol@alexandra.dk

