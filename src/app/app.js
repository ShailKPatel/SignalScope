// SignalScope web client - Team Takneeki Vibhag
// SIH 2026 Internal Hackathon, Problem Statement 2 (C-433)
//
// Static text lives in index.html with data-i18n keys. English is read from the
// page at load, so HI_STATIC only holds Hindi overrides. UI holds strings built
// at runtime (verdicts, table cells) in both languages.

let currentResultData = null;
let currentBatchData = null;
let currentBatchFiles = [];
let activeMode = 'single'; // 'single' | 'folder' | 'eval' | 'about' | 'team'
let currentLang = 'en';
let isProcessing = false;

const EN_STATIC = {};

const HI_STATIC = {
  modalTitle: "सिग्नलस्कोप के बारे में",
  modalHead: "वास्तविक बनाम एआई-जनित छवि पहचान",
  modalDesc: "सिग्नलस्कोप अनुमान लगाता है कि कोई छवि एआई-जनित होने की कितनी संभावना है, उस अनुमान के पीछे के संकेत दिखाता है, और फ़ाइल के उत्पत्ति मेटाडेटा की जांच करता है। यह दृश्य, वस्तुएं, कला और उत्पाद छवियों के लिए है।",
  modalNote: "हर परिणाम एक संभावना आकलन है, प्रमाण या आरोप नहीं। यह टूल वास्तविक व्यक्तियों की पहचान नहीं करता और न ही उनके बारे में दावे करता है।",
  modalBtn: "आगे बढ़ें",
  skipLink: "मुख्य सामग्री पर जाएं",
  utilTitle: "एसआईएच 2026 आंतरिक हैकाथॉन · समस्या कथन 2 (C-433)",
  textSize: "पाठ आकार:",
  btnContrast: "उच्च कंट्रास्ट",
  hdrKicker: "टीम तकनीकी विभाग",
  hdrTitle: "सिग्नल स्कोप",
  hdrSub: "जनरेटिव मीडिया के युग में वास्तविक और सिंथेटिक की पहचान",
  navSingle: "छवि विश्लेषण",
  navBatch: "बैच फ़ोल्डर स्कैन",
  navEval: "मूल्यांकन एवं मेट्रिक्स",
  navMethod: "कार्यप्रणाली",
  navTeam: "टीम (6)",
  tickerLabel: "परिणाम कैसे पढ़ें",
  tickerText: "स्कोर कैलिब्रेटेड संभावनाएं हैं (\"संभवतः एआई-जनित\" / \"संभवतः वास्तविक\"), प्रमाण नहीं। CIFAKE टेस्ट स्प्लिट: ROC-AUC 0.9976, सटीकता 97.8%, फॉल्स-पॉजिटिव दर 2.35%।",
  tilesTitle: "सिग्नलस्कोप क्या करता है",
  tile1T: "एकल छवि विश्लेषण",
  tile1D: "स्टैक्ड 3-मॉडल एन्सेम्बल निर्णय, सैलिएंसी ओवरले, मापे गए स्पष्टीकरण संकेत।",
  tile2T: "बैच फ़ोल्डर स्कैन",
  tile2D: "पूरे फ़ोल्डर की छवियों को स्कैन करें और किसी भी परिणाम को विस्तार से देखें।",
  tile3T: "मूल्यांकन एवं मेट्रिक्स",
  tile3D: "20,000 CIFAKE टेस्ट छवियों पर ROC-AUC, मैक्रो-F1, कन्फ्यूजन मैट्रिक्स और ऑपरेटिंग पॉइंट।",
  tile4T: "कार्यप्रणाली",
  tile4D: "आर्किटेक्चर, प्रशिक्षण सेटअप और इस सबमिशन में शामिल मॉड्यूल।",
  crumbHome: "होम",
  singleTitle: "छवि विश्लेषण",
  dropTitle: "छवि चुनें या खींचकर छोड़ें",
  dropSub: "JPEG, PNG या WebP। आपको संभावना निर्णय, एन्सेम्बल विवरण, मापे गए संकेतों के साथ सैलिएंसी ओवरले, क्षरण पुनः-परीक्षण और उत्पत्ति मेटाडेटा जांच मिलती है।",
  btnChoose: "छवि फ़ाइल चुनें",
  batchTitle: "बैच फ़ोल्डर स्कैन",
  batchDropTitle: "छवियों का फ़ोल्डर चुनें",
  batchDropSub: "फ़ोल्डर की हर छवि को स्कोर किया जाता है। आपको सारांश और तालिका मिलती है, और किसी भी छवि का पूरा विश्लेषण खोल सकते हैं।",
  btnChooseDir: "छवि फ़ोल्डर चुनें",
  evalTitle: "मूल्यांकन एवं मेट्रिक्स",
  evalTag: "CIFAKE टेस्ट स्प्लिट · 20,000 छवियां",
  evalIntro: "प्राथमिक मेट्रिक: ROC-AUC। मुख्य आंकड़े पूर्ण CIFAKE टेस्ट स्प्लिट (10,000 वास्तविक, 10,000 एआई-जनित) पर डिफ़ॉल्ट 0.5 सीमा पर तैनात स्टैक्ड एन्सेम्बल के हैं।",
  statAucL: "ROC-AUC (प्राथमिक)",
  statF1L: "मैक्रो-F1",
  statAccL: "सटीकता @ 0.5",
  statFprL: "फॉल्स-पॉजिटिव दर @ 0.5",
  evalDataT: "डेटासेट एवं विभाजन",
  thSplit: "विभाजन",
  thImages: "छवियां",
  thRealFake: "वास्तविक / एआई",
  thUse: "उपयोग",
  splitTrain: "ट्रेन (CIFAKE train/ का 90%)",
  splitTrainUse: "डुअल-स्ट्रीम मॉडल वेट्स",
  splitVal: "वैलिडेशन (CIFAKE train/ का 10%)",
  splitValUse: "सर्वश्रेष्ठ epoch, तापमान कैलिब्रेशन, स्टैकिंग मेटा-लर्नर (4,000 छवियों का संतुलित नमूना)",
  splitTest: "टेस्ट (CIFAKE test/)",
  splitTestUse: "सभी रिपोर्ट किए गए मेट्रिक्स",
  dataSource: "स्रोत: CIFAKE (birdy654/cifake-real-and-ai-generated-synthetic-images, Kaggle)। वास्तविक छवियां CIFAR-10 से; एआई छवियां Stable Diffusion v1.4 से; सभी 32×32।",
  evalCmpT: "टेस्ट स्प्लिट पर हर मॉडल",
  thModel: "मॉडल",
  thAcc: "सटीकता",
  thPrec: "प्रिसिजन",
  thRecall: "रिकॉल",
  rowMajority: "बहुमत मतदान (3 सदस्य)",
  rowStacked: "स्टैक्ड एन्सेम्बल @ 0.5 (तैनात)",
  rowStackedLow: "स्टैक्ड एन्सेम्बल @ 0.165 (5% वैलिडेशन-FPR बिंदु)",
  evalCmpNote: "पॉजिटिव क्लास = एआई-जनित। FPR = वास्तविक छवियों का वह हिस्सा जिसे गलती से एआई बताया गया।",
  evalCmT: "कन्फ्यूजन मैट्रिक्स (टेस्ट स्प्लिट)",
  cmStacked: "स्टैक्ड एन्सेम्बल @ 0.5 · मैक्रो-F1 0.9780",
  cmDual: "केवल डुअल-स्ट्रीम @ 0.5 · मैक्रो-F1 0.9774",
  cmPredReal: "अनुमानित वास्तविक",
  cmPredAi: "अनुमानित एआई",
  cmActReal: "वास्तविक में असली",
  cmActAi: "वास्तव में एआई",
  evalOpT: "ऑपरेटिंग पॉइंट एवं कैलिब्रेशन",
  op1: "डिफ़ॉल्ट सीमा 0.5: सटीकता 97.80%, FPR 2.35%।",
  op2: "कम-चूक विकल्प 0.165 (~5% FPR के लिए वैलिडेशन पर चुना गया): रिकॉल 99.19%, FPR 5.71%।",
  op3: "डुअल-स्ट्रीम विश्वास को वैलिडेशन पर तापमान स्केलिंग से कैलिब्रेट किया गया (T = 1.066)।",
  op4: "स्कोर संभावना के रूप में दिखाए जाते हैं; किसी वास्तविक फोटो को गलत चिह्नित करना महंगी त्रुटि मानी जाती है।",
  evalStackT: "स्टैकिंग मेटा-लर्नर",
  st1: "हर सदस्य के logit(P(AI)) पर L2 (रिज) लॉजिस्टिक रिग्रेशन; C = 31.6, 5-फोल्ड CV से चुना गया।",
  st2: "वैलिडेशन पर आउट-ऑफ-फोल्ड: ROC-AUC 0.9977, सटीकता 97.80%।",
  st3: "सीखे गए वेट: डुअल-स्ट्रीम +7.53, ViT −0.29, Swin +0.17।",
  st4: "यह उन सदस्यों का वेट घटाकर जो CIFAKE पर काम नहीं करते, सटीकता 63.8% (बहुमत मतदान) से 97.8% तक बढ़ाता है।",
  evalTrainT: "प्रशिक्षण सेटअप (डुअल-स्ट्रीम सदस्य)",
  tr1: "बैकबोन: ResNet34 (ImageNet-प्रीट्रेन्ड) + 2D-FFT फ्रीक्वेंसी शाखा",
  tr2: "मूल 32×32 इनपुट, कोई अपसैंपलिंग नहीं",
  tr3: "6 epoch, बैच 256, AdamW (lr 3e-4, wd 0.01), कोसाइन शेड्यूल, मिक्स्ड प्रिसिजन",
  tr4: "ऑगमेंटेशन: रैंडम JPEG पुनः-संपीड़न (Q50–95), रिफ्लेक्ट-पैड क्रॉप, फ्लिप, रंग जिटर",
  tr5: "सर्वश्रेष्ठ epoch वैलिडेशन ROC-AUC पर चुना गया",
  tr6: "Kaggle GPU पर लगभग 20 मिनट में प्रशिक्षित",
  evalUnseenT: "अनदेखे जनरेटर पर प्रदर्शन",
  evalUnseenD: "CIFAKE में केवल एक जनरेटर (Stable Diffusion v1.4) है, इसलिए इसमें अनदेखे-जनरेटर का विभाजन नहीं है और हम वह AUC स्वयं नहीं माप सकते। आयोजक अपने अलग रखे गए सेट पर हमारे predict इंटरफ़ेस से कुल और अनदेखे-जनरेटर AUC की गणना करते हैं।",
  evalLimT: "ज्ञात सीमाएं",
  lim1: "एक ही जनरेटर पर प्रशिक्षित: नए या अनदेखे जनरेटरों पर सटीकता मापी नहीं गई है और काफी कम हो सकती है।",
  lim2: "डुअल-स्ट्रीम सदस्य छवियों को 32×32 पर देखता है, इसलिए बड़ी छवियों के सूक्ष्म उच्च-रिज़ॉल्यूशन आर्टिफैक्ट छूट जाते हैं।",
  lim3: "ViT और Swin CIFAKE पर लगभग अनुमान-स्तर पर हैं (AUC 0.41 और 0.65), इसलिए CIFAKE पर फिट स्टैकर लगभग पूरी तरह डुअल-स्ट्रीम मॉडल पर निर्भर है।",
  lim4: "सैलिएंसी ओवरले दिखाता है कि डुअल-स्ट्रीम मॉडल का स्कोर किन पिक्सेल के प्रति सबसे संवेदनशील है; यह आर्टिफैक्ट का सत्यापित स्थानीयकरण नहीं है।",
  lim5: "मेटाडेटा हस्ताक्षर हटाए या नकली बनाए जा सकते हैं; घोषित जनरेटर केवल मेटाडेटा साक्ष्य के रूप में बताया जाता है।",
  methodTitle: "कार्यप्रणाली",
  mModulesT: "इस सबमिशन के मॉड्यूल",
  chipCore: "कोर: वास्तविक बनाम एआई क्लासिफायर",
  chipA: "A: स्पष्टीकरण",
  chipC: "C: मजबूती",
  chipD: "D: उत्पत्ति एवं मेटाडेटा",
  chipF: "F: तैनात करने योग्य वेब ऐप",
  mNotD: "इस सबमिशन में शामिल नहीं: B (जनरेटर एट्रिब्यूशन), E (छवि-पाठ संगति), G (सक्रिय रक्षा)।",
  mPipeT: "पाइपलाइन",
  mPipeD: "छवि → स्तर 1: उत्पत्ति मेटाडेटा जांच (स्पष्ट एआई-जनरेटर हस्ताक्षर या C2PA मेनिफेस्ट) → स्तर 2: स्टैक्ड पिक्सेल एन्सेम्बल → कैलिब्रेटेड संभावना → स्पष्टीकरण संकेत, सैलिएंसी ओवरले और क्षरण पुनः-परीक्षण → निर्णय \"संभवतः एआई-जनित\" या \"संभवतः वास्तविक\"।",
  mCoreT: "कोर: स्टैक्ड 3-मॉडल एन्सेम्बल",
  mCoreD: "सदस्य: ViT-Base (dima806/deepfake_vs_real_image_detection), Swin (Organika/sdxl-detector), और मूल 32×32 पर CIFAKE पर प्रशिक्षित हमारा ResNet34 + 2D-FFT डुअल-स्ट्रीम मॉडल। हर सदस्य का P(AI) CIFAKE वैलिडेशन छवियों पर फिट रिज लॉजिस्टिक स्टैकिंग मेटा-लर्नर से जोड़ा जाता है।",
  mAT: "मॉड्यूल A: स्पष्टीकरण",
  mAD: "डुअल-स्ट्रीम मॉडल (जो निर्णय को मुख्य रूप से चलाता है) का SmoothGrad सैलिएंसी मैप उसके 32×32 इनपुट पर गणना कर छवि पर ओवरले करता है। एक डिलीशन जांच परखती है कि शीर्ष 10% सैलिएंट पिक्सेल छिपाने से स्कोर यादृच्छिक पिक्सेल छिपाने की तुलना में अधिक बदलता है या नहीं। संकेत: हर सदस्य का P(AI), स्टैक्ड log-odds में उसका हिस्सा, छवि की उच्च-आवृत्ति स्पेक्ट्रल ऊर्जा, और कोई भी मेटाडेटा हस्ताक्षर। सैलिएंसी संवेदनशीलता दिखाती है, आर्टिफैक्ट का सत्यापित स्थानीयकरण नहीं।",
  mCT: "मॉड्यूल C: क्षरण के प्रति मजबूती",
  mCD: "अपलोड की गई छवि को JPEG गुणवत्ता 90/70/50/30 पर पुनः एन्कोड और 75/50/25% तक छोटा करता है, मूल और हर संस्करण को उसी पिक्सेल एन्सेम्बल से दोबारा स्कोर करता है, और बताता है कि हर चरण पर निर्णय बना रहा या नहीं।",
  mDT: "मॉड्यूल D: उत्पत्ति एवं मेटाडेटा",
  mDD: "EXIF, XMP और PNG टेक्स्ट फ़ील्ड, और मौजूद होने पर C2PA कंटेंट क्रेडेंशियल्स पढ़ता है। स्पष्ट एआई-जनरेटर हस्ताक्षर मेटाडेटा साक्ष्य के रूप में बताया जाता है और स्तर 1 पर निर्णय करता है; अन्यथा पिक्सेल एन्सेम्बल निर्णय करता है।",
  mFT: "मॉड्यूल F: तैनात करने योग्य वेब ऐप",
  mFD: "FastAPI बैकएंड (एकल-छवि और बैच एंडपॉइंट) के साथ यह द्विभाषी डैशबोर्ड: ड्रैग-एंड-ड्रॉप अपलोड, फ़ोल्डर स्कैन, समायोज्य ओवरले, पाठ-आकार और उच्च-कंट्रास्ट नियंत्रण।",
  teamTitle: "टीम तकनीकी विभाग",
  teamSub: "एसआईएच 2026 आंतरिक हैकाथॉन · समस्या कथन 2 (C-433) · टीम लीडर: राजवी चौहान · 6/6 सदस्य · पंजीकृत",
  cardLbl: "टीम सदस्य",
  cardLblLead: "टीम लीडर",
  m1Badge: "टीम लीड",
  m1Name: "राजवी चौहान",
  m1Desig: "टीम लीडर एवं एआई अनुसंधान प्रमुख",
  m1Role: "टीम समन्वय, मॉडल प्रशिक्षण, डेटा विभाजन और फ्रीक्वेंसी-डोमेन फीचर्स।",
  m2Badge: "मॉड्यूल C एवं D",
  m2Name: "दिव्येश प्रजापति",
  m2Desig: "उत्पत्ति एवं मजबूती",
  m2Role: "मॉड्यूल D EXIF/C2PA मेटाडेटा पार्सर और मॉड्यूल C क्षरण पुनः-परीक्षण।",
  m3Badge: "मॉड्यूल F",
  m3Name: "सर्वेश मुदलियार",
  m3Desig: "फुल-स्टैक इंजीनियर",
  m3Role: "मॉड्यूल F वेब डैशबोर्ड, सुलभता सुविधाएं और FastAPI एकीकरण।",
  m4Badge: "अनुसंधान",
  m4Name: "शाह हेनिल संदीपकुमार",
  m4Desig: "मल्टीमॉडल एवं मूल्यांकन",
  m4Role: "मल्टीमॉडल छवि-पाठ अनुसंधान और मूल्यांकन रिपोर्टिंग।",
  m5Badge: "एआई कोर",
  m5Name: "शैल किरण पटेल",
  m5Desig: "सिस्टम आर्किटेक्ट",
  m5Role: "कोर आर्किटेक्चर, स्टैक्ड-एन्सेम्बल प्रशिक्षण पाइपलाइन और तैनाती।",
  m6Badge: "मॉड्यूल A",
  m6Name: "कसक गोहिल",
  m6Desig: "कंप्यूटर विजन एवं स्पष्टीकरण",
  m6Role: "मॉड्यूल A सैलिएंसी ओवरले और मापे गए स्पष्टीकरण संकेत।",
  certHeader: "संभावना आकलन",
  heatmapLabel: "सैलिएंसी ओवरले (डुअल-स्ट्रीम मॉडल)",
  ensT: "निर्णय कैसे लिया गया",
  tabA: "A · स्पष्टीकरण",
  tabC: "C · मजबूती",
  tabD: "D · उत्पत्ति",
  headA: "निर्णय के पीछे मापे गए संकेत",
  headC: "JPEG संपीड़न और छोटा करने पर निर्णय",
  lblPixelVerdict: "मूल छवि पर पिक्सेल डिटेक्टर (मेटाडेटा अनदेखा)",
  lblRating: "स्थिरता",
  lblJpeg70: "JPEG Q70 पर निर्णय बना रहा",
  lblResize50: "50% आकार पर निर्णय बना रहा",
  lblJpegCurve: "JPEG पुनः-संपीड़न (एआई-संभावना स्कोर)",
  lblResizeCurve: "आकार घटाना (एआई-संभावना स्कोर)",
  headD: "उत्पत्ति एवं फ़ाइल मेटाडेटा",
  lblExif: "EXIF मेटाडेटा",
  lblC2pa: "C2PA कंटेंट क्रेडेंशियल्स",
  lblCamera: "कैमरा रिकॉर्ड",
  lblDeclaredGen: "मेटाडेटा में घोषित जनरेटर",
  lblMetaAssess: "आकलन",
  batchResT: "बैच स्कैन परिणाम",
  statTotal: "स्कैन की गई छवियां",
  statAi: "संभवतः एआई-जनित",
  statReal: "संभवतः वास्तविक",
  statLatency: "कुल समय",
  thFile: "फ़ाइल",
  thVerdict: "निर्णय",
  thScore: "एआई-संभावना",
  thLevel: "निर्णय का आधार",
  thAction: "विवरण",
  footerText: "सिग्नलस्कोप · टीम तकनीकी विभाग · एसआईएच 2026 आंतरिक हैकाथॉन, समस्या कथन 2 (C-433) · शैक्षणिक हैकाथॉन परियोजना, भारत सरकार से संबद्ध नहीं।"
};

const UI = {
  en: {
    crumbSingle: "Analyse an Image",
    crumbFolder: "Batch Folder Scan",
    crumbEval: "Evaluation & Metrics",
    crumbAbout: "Methodology",
    crumbTeam: "Team",
    verdictAi: "LIKELY AI-GENERATED",
    verdictReal: "LIKELY REAL",
    scoreSub: "AI-likelihood score",
    level1: "Metadata signature",
    level2: "Pixel ensemble",
    analysing: "Analysing image…",
    analysingSub: name => `Running the detector on ${name}. Please wait.`,
    analysingBatch: n => `Analysing ${n} images…`,
    analysingBatchSub: "Large folders can take a while. Please wait.",
    invalidFile: "Please select a valid image file (JPEG, PNG, WebP).",
    noImages: "No image files found in the selected folder.",
    apiError: "The SignalScope backend could not analyse this input: ",
    inspect: "Open",
    present: "Present",
    absent: "Not found",
    signed: "Manifest found",
    unsigned: "No manifest",
    unknown: "Not recorded",
    stable: "Kept",
    changed: "Changed",
    na: "N/A",
    notDeclared: "None declared",
    original: "Original",
    quality: "JPEG Q",
    member: "Member",
    contrib: "share of decision",
    stackedP: "Stacked P(AI)",
    threshold: "threshold",
    fusion: "Fusion",
    votes: (a, n) => `${a} of ${n} members lean AI`,
    opPoint: (acc, fpr) => `At this threshold on the CIFAKE test split: accuracy ${acc}, false-positive rate ${fpr}.`,
    level1Note: gen => `Decided at Level 1: the file's metadata declares an AI generator (${gen}). The pixel ensemble was not needed for this verdict.`,
    noEnsemble: "No ensemble breakdown is available for this result.",
    robustUnavailable: "Robustness re-test unavailable for this image."
  },
  hi: {
    crumbSingle: "छवि विश्लेषण",
    crumbFolder: "बैच फ़ोल्डर स्कैन",
    crumbEval: "मूल्यांकन एवं मेट्रिक्स",
    crumbAbout: "कार्यप्रणाली",
    crumbTeam: "टीम",
    verdictAi: "संभवतः एआई-जनित",
    verdictReal: "संभवतः वास्तविक",
    scoreSub: "एआई-संभावना स्कोर",
    level1: "मेटाडेटा हस्ताक्षर",
    level2: "पिक्सेल एन्सेम्बल",
    analysing: "छवि का विश्लेषण हो रहा है…",
    analysingSub: name => `${name} पर डिटेक्टर चल रहा है। कृपया प्रतीक्षा करें।`,
    analysingBatch: n => `${n} छवियों का विश्लेषण हो रहा है…`,
    analysingBatchSub: "बड़े फ़ोल्डर में कुछ समय लग सकता है। कृपया प्रतीक्षा करें।",
    invalidFile: "कृपया एक वैध छवि फ़ाइल (JPEG, PNG, WebP) चुनें।",
    noImages: "चयनित फ़ोल्डर में कोई छवि फ़ाइल नहीं मिली।",
    apiError: "सिग्नलस्कोप बैकएंड इस इनपुट का विश्लेषण नहीं कर सका: ",
    inspect: "खोलें",
    present: "मौजूद",
    absent: "नहीं मिला",
    signed: "मेनिफेस्ट मिला",
    unsigned: "कोई मेनिफेस्ट नहीं",
    unknown: "दर्ज नहीं",
    stable: "बना रहा",
    changed: "बदला",
    na: "लागू नहीं",
    notDeclared: "कोई घोषित नहीं",
    original: "मूल",
    quality: "JPEG Q",
    member: "सदस्य",
    contrib: "निर्णय में हिस्सा",
    stackedP: "स्टैक्ड P(AI)",
    threshold: "सीमा",
    fusion: "संयोजन",
    votes: (a, n) => `${n} में से ${a} सदस्य एआई की ओर`,
    opPoint: (acc, fpr) => `CIFAKE टेस्ट स्प्लिट पर इस सीमा पर: सटीकता ${acc}, फॉल्स-पॉजिटिव दर ${fpr}।`,
    level1Note: gen => `स्तर 1 पर निर्णय: फ़ाइल का मेटाडेटा एक एआई जनरेटर (${gen}) घोषित करता है। इस निर्णय के लिए पिक्सेल एन्सेम्बल की आवश्यकता नहीं थी।`,
    noEnsemble: "इस परिणाम के लिए कोई एन्सेम्बल विवरण उपलब्ध नहीं है।",
    robustUnavailable: "इस छवि के लिए मजबूती पुनः-परीक्षण उपलब्ध नहीं है।"
  }
};

function t(key, ...args) {
  const entry = (UI[currentLang] && UI[currentLang][key]) ?? UI.en[key];
  return typeof entry === 'function' ? entry(...args) : (entry ?? key);
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Probabilities never display as a flat 0% or 100%: they are likelihoods, not certainty.
function pct(x, digits = 1) {
  if (typeof x !== 'number') return t('na');
  if (x >= 0.999) return '>99.9%';
  if (x <= 0.001) return '<0.1%';
  return `${(x * 100).toFixed(digits)}%`;
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    if (!(el.dataset.i18n in EN_STATIC)) EN_STATIC[el.dataset.i18n] = el.textContent;
  });
  setupDragAndDrop();
  checkHashRoute();
  window.addEventListener('hashchange', checkHashRoute);
});

function closeModal() {
  const modal = document.getElementById('advisoryModal');
  if (modal) modal.style.display = 'none';
}

function adjustFontSize(delta) {
  document.body.classList.remove('font-sm', 'font-lg');
  if (delta === -1) document.body.classList.add('font-sm');
  else if (delta === 1) document.body.classList.add('font-lg');
}

function toggleContrast() {
  document.body.classList.toggle('contrast-high');
}

function checkHashRoute() {
  const routes = {
    '#single': 'single', '#folder': 'folder', '#batch': 'folder',
    '#eval': 'eval', '#metrics': 'eval', '#evaluation': 'eval',
    '#about': 'about', '#methodology': 'about', '#team': 'team'
  };
  const mode = routes[window.location.hash.toLowerCase()];
  if (mode) switchMode(mode);
}

function setLanguage(lang) {
  currentLang = lang;
  document.documentElement.lang = lang;
  document.getElementById('btnLangEN')?.classList.toggle('active', lang === 'en');
  document.getElementById('btnLangHI')?.classList.toggle('active', lang === 'hi');

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    el.textContent = (lang === 'hi' && HI_STATIC[key]) || EN_STATIC[key] || el.textContent;
  });
  updateBreadcrumbText();

  if (currentResultData) renderSingleResult(currentResultData);
  if (currentBatchData) renderBatchResult(currentBatchData);
}

function updateBreadcrumbText() {
  const keys = { single: 'crumbSingle', folder: 'crumbFolder', eval: 'crumbEval', about: 'crumbAbout', team: 'crumbTeam' };
  setText('txtBreadcrumbCurrent', t(keys[activeMode]));
}

function switchMode(mode) {
  activeMode = mode;
  const sections = {
    single: 'dropZoneSingle', folder: 'dropZoneFolder', eval: 'evalSection', about: 'aboutSection', team: 'teamSection'
  };
  const buttons = {
    single: 'btnMenuSingle', folder: 'btnMenuFolder', eval: 'btnMenuEval', about: 'btnMenuAbout', team: 'btnMenuTeam'
  };

  Object.entries(sections).forEach(([m, id]) => {
    const el = document.getElementById(id);
    if (el) el.style.display = m === mode ? 'block' : 'none';
  });
  Object.entries(buttons).forEach(([m, id]) => {
    document.getElementById(id)?.classList.toggle('active', m === mode);
  });

  const resultsGrid = document.getElementById('resultsGrid');
  if (resultsGrid) resultsGrid.style.display = mode === 'single' && currentResultData ? 'grid' : 'none';
  const batchSection = document.getElementById('batchSection');
  if (batchSection) batchSection.style.display = mode === 'folder' && currentBatchData ? 'block' : 'none';

  updateBreadcrumbText();
}

function setupDragAndDrop() {
  const setupZone = (dropZone, isFolder) => {
    if (!dropZone) return;
    ['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    }));
    ['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    }));
    dropZone.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (!files.length) return;
      if (isFolder || files.length > 1) {
        switchMode('folder');
        processFolderFiles(files);
      } else {
        switchMode('single');
        processSingleFile(files[0]);
      }
    });
  };
  setupZone(document.getElementById('dropZoneSingle'), false);
  setupZone(document.getElementById('dropZoneFolder'), true);
}

function handleSingleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length) processSingleFile(files[0]);
}

function handleFolderSelect(event) {
  const files = event.target.files;
  if (files && files.length) processFolderFiles(files);
}

async function processSingleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert(t('invalidFile'));
    return;
  }
  await uploadSingleFile(file);
}

async function processFolderFiles(fileList) {
  const files = Array.from(fileList).filter(f => f.type.startsWith('image/'));
  if (!files.length) {
    alert(t('noImages'));
    return;
  }
  await uploadBatchFiles(files);
}

function setProcessing(cardId, active, title, sub) {
  const zone = document.querySelector(`#${cardId} .gov-dropzone`);
  if (!zone) return;
  let overlay = zone.querySelector('.processing-overlay');
  if (!active) {
    overlay?.remove();
    zone.removeAttribute('aria-busy');
    return;
  }
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'processing-overlay';
    overlay.setAttribute('role', 'status');
    overlay.setAttribute('aria-live', 'polite');
    overlay.innerHTML =
      '<div class="processing-spinner"></div>' +
      '<div class="processing-title"></div>' +
      '<div class="processing-bar"></div>' +
      '<div class="processing-sub"></div>';
    zone.appendChild(overlay);
  }
  overlay.querySelector('.processing-title').textContent = title;
  overlay.querySelector('.processing-sub').textContent = sub || '';
  zone.setAttribute('aria-busy', 'true');
}

async function readError(resp) {
  try {
    const body = await resp.json();
    return body.detail || `HTTP ${resp.status}`;
  } catch {
    return `HTTP ${resp.status}`;
  }
}

async function uploadSingleFile(file) {
  if (isProcessing) return;
  isProcessing = true;
  setProcessing('dropZoneSingle', true, t('analysing'), t('analysingSub', file.name));

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/api/predict', { method: 'POST', body: formData });
    if (!resp.ok) throw new Error(await readError(resp));
    const data = await resp.json();
    setBaseImage(file);
    currentResultData = data;
    renderSingleResult(data);
  } catch (err) {
    console.error('Error analysing image:', err);
    alert(t('apiError') + err.message);
  } finally {
    isProcessing = false;
    setProcessing('dropZoneSingle', false);
    const input = document.getElementById('fileInputSingle');
    if (input) input.value = '';
  }
}

async function uploadBatchFiles(files) {
  if (isProcessing) return;
  isProcessing = true;
  setProcessing('dropZoneFolder', true, t('analysingBatch', files.length), t('analysingBatchSub'));

  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  try {
    const resp = await fetch('/api/predict-batch', { method: 'POST', body: formData });
    if (!resp.ok) throw new Error(await readError(resp));
    currentBatchFiles = files;
    currentBatchData = await resp.json();
    renderBatchResult(currentBatchData);
  } catch (err) {
    console.error('Error processing batch:', err);
    alert(t('apiError') + err.message);
  } finally {
    isProcessing = false;
    setProcessing('dropZoneFolder', false);
    const input = document.getElementById('folderInputDirectory');
    if (input) input.value = '';
  }
}

function setBaseImage(file) {
  const baseImg = document.getElementById('baseImage');
  if (!baseImg || !file) return;
  if (baseImg.dataset.objectUrl) URL.revokeObjectURL(baseImg.dataset.objectUrl);
  baseImg.dataset.objectUrl = URL.createObjectURL(file);
  baseImg.src = baseImg.dataset.objectUrl;
}

function levelLabel(verdict) {
  return String(verdict.detection_level || '').startsWith('Level 1') ? t('level1') : t('level2');
}

function renderSingleResult(data) {
  const resultsGrid = document.getElementById('resultsGrid');
  if (resultsGrid) resultsGrid.style.display = 'grid';

  const verdict = data.verdict || {};
  const modules = data.modules || {};
  const isAi = Boolean(verdict.is_ai_generated);

  setText('verdictLabel', isAi ? t('verdictAi') : t('verdictReal'));
  setText('verdictScore', pct(verdict.confidence_score));
  setText('verdictSubtitle', `${t('scoreSub')} · ${levelLabel(verdict)}`);
  setText('resultFilename', data.filename || '');
  setText('resultTiming', typeof data.processing_time_ms === 'number' ? `${(data.processing_time_ms / 1000).toFixed(1)} s` : '');
  setText('imageResolution', data.resolution || '');
  const banner = document.getElementById('verdictBanner');
  if (banner) banner.className = `verdict-banner-gov ${isAi ? 'ai' : 'real'}`;

  renderExplanation(modules.module_a_explanation || modules.module_a_explainability || {});
  renderEnsemble(verdict);
  renderRobustness(modules.module_c_robustness);
  renderMetadata(modules.module_d_metadata || {}, modules.module_b_attribution || {});
}

function renderExplanation(modA) {
  const heatmapImg = document.getElementById('heatmapImage');
  if (heatmapImg) {
    heatmapImg.src = modA.heatmap_base64 || '';
    heatmapImg.hidden = !modA.heatmap_base64;
  }
  setText('saliencyNote', modA.localization_quality || '');
  setText('explanationSummary', modA.summary_text || '');

  const cues = document.getElementById('visualCuesContainer');
  if (cues) {
    cues.innerHTML = (modA.cues || []).map(c => `
      <div class="cue-item-gov">
        <div class="cue-header-gov">
          <span>${esc(c.type)}</span>
          <span class="cue-value">${esc(c.value)}</span>
        </div>
        <div class="cue-desc-gov">${esc(c.detail)}</div>
      </div>
    `).join('');
  }
}

function renderEnsemble(verdict) {
  const box = document.getElementById('ensembleContainer');
  if (!box) return;

  if (String(verdict.detection_level || '').startsWith('Level 1')) {
    box.innerHTML = `<p class="note-muted">${esc(t('level1Note', verdict.matched_generator || '-'))}</p>`;
    return;
  }

  const ens = verdict.ensemble_breakdown;
  if (!ens || !Array.isArray(ens.family_models)) {
    box.innerHTML = `<p class="note-muted">${esc(t('noEnsemble'))}</p>`;
    return;
  }

  const stacking = ens.stacking || {};
  const contributions = stacking.member_logit_contributions || {};
  const rows = ens.family_models.map(m => {
    const c = contributions[m.model_id];
    const share = typeof c === 'number' ? `<span class="contrib ${c > 0 ? 'toward-ai' : 'toward-real'}">${c > 0 ? '+' : ''}${c.toFixed(2)} ${esc(t('contrib'))}</span>` : '';
    return `
      <div class="metric-row-gov">
        <span class="metric-label-gov">${esc(m.family)}<br><span class="member-sub">${esc(m.model_name)}</span></span>
        <span class="metric-val-gov">P(AI) ${pct(m.ai_probability)} ${share}</span>
      </div>`;
  }).join('');

  let fused = '';
  if (typeof stacking.stacked_ai_probability === 'number') {
    fused = `
      <div class="metric-row-gov metric-row-strong">
        <span class="metric-label-gov">${esc(t('stackedP'))}</span>
        <span class="metric-val-gov">${pct(stacking.stacked_ai_probability)} (${esc(t('threshold'))} ${Number(stacking.decision_threshold ?? 0.5).toFixed(2)})</span>
      </div>`;
  } else if (typeof ens.ai_votes === 'number') {
    fused = `
      <div class="metric-row-gov metric-row-strong">
        <span class="metric-label-gov">${esc(t('fusion'))}</span>
        <span class="metric-val-gov">${esc(t('votes', ens.ai_votes, ens.num_families_evaluated))}</span>
      </div>`;
  }

  const op = typeof verdict.operating_point_accuracy === 'number' && typeof verdict.operating_point_fpr === 'number'
    ? `<p class="note-muted">${esc(t('opPoint', pct(verdict.operating_point_accuracy, 2), pct(verdict.operating_point_fpr, 2)))}</p>`
    : '';

  box.innerHTML = rows + fused + op;
}

function stablePill(ok) {
  if (ok === null || ok === undefined) return esc(t('na'));
  return `<span class="pill-gov ${ok ? 'pill-gov-real' : 'pill-gov-ai'}">${esc(ok ? t('stable') : t('changed'))}</span>`;
}

function curveRows(rows, labelFn) {
  return rows.map(r => `
    <div class="metric-row-gov">
      <span class="metric-label-gov">${esc(labelFn(r))}</span>
      <span class="metric-val-gov">${pct(r.ai_score)} ${stablePill(r.verdict_stable)}</span>
    </div>
  `).join('');
}

function renderRobustness(modC) {
  const jpeg = document.getElementById('jpegCurveContainer');
  const resize = document.getElementById('resizeCurveContainer');

  if (!modC || modC.status !== 'measured') {
    setText('robustnessPixel', t('na'));
    setText('robustnessRating', modC?.overall_stability_rating || t('robustUnavailable'));
    document.getElementById('robustnessJpeg').innerHTML = esc(t('na'));
    document.getElementById('robustnessResize').innerHTML = esc(t('na'));
    if (jpeg) jpeg.innerHTML = '';
    if (resize) resize.innerHTML = '';
    setText('robustnessMethod', '');
    return;
  }

  // On metadata (Level 1) verdicts this is the pixel ensemble's own call and can differ from the headline verdict.
  const pixelAi = modC.pixel_detector_clean_score >= 0.5;
  setText('robustnessPixel', `${pixelAi ? t('verdictAi') : t('verdictReal')} · ${pct(modC.pixel_detector_clean_score)}`);
  setText('robustnessRating', modC.overall_stability_rating);
  document.getElementById('robustnessJpeg').innerHTML = stablePill(modC.verdict_preserved_under_jpeg70);
  document.getElementById('robustnessResize').innerHTML = stablePill(modC.verdict_preserved_under_resize50);
  if (jpeg) {
    jpeg.innerHTML = curveRows(modC.jpeg_degradation_curve || [],
      r => (r.quality === 'Original' ? t('original') : `${t('quality')}${r.quality}`));
  }
  if (resize) {
    resize.innerHTML = curveRows(modC.resize_degradation_curve || [],
      r => String(r.resolution).replace(/^Original/, t('original')));
  }
  setText('robustnessMethod', modC.method || '');
}

function renderMetadata(modD, modB) {
  setText('metaHasExif', modD.has_exif ? t('present') : t('absent'));
  setText('metaHasC2pa', modD.has_c2pa ? t('signed') : t('unsigned'));
  setText('metaCamera', modD.camera_model ? `${modD.camera_make || ''} ${modD.camera_model}`.trim() : t('unknown'));
  setText('metaGenerator', modB.status === 'metadata_only' ? modB.family : t('notDeclared'));
  setText('metaAssessment', modD.assessment || t('na'));
}

function renderBatchResult(batchData) {
  const batchSection = document.getElementById('batchSection');
  if (batchSection) batchSection.style.display = 'block';

  const summary = batchData.summary || {};
  setText('statTotalImages', batchData.total_images ?? 0);
  setText('statAiCount', summary.likely_ai_generated_count ?? 0);
  setText('statRealCount', summary.likely_real_count ?? 0);
  setText('statLatency', `${batchData.total_time_seconds ?? 0}s`);

  const tbody = document.getElementById('batchTableBody');
  if (!tbody) return;
  tbody.innerHTML = (batchData.results || []).map((item, idx) => {
    const v = item.verdict || {};
    const isAi = Boolean(v.is_ai_generated);
    return `
      <tr>
        <td>${esc(item.filename)}</td>
        <td><span class="pill-gov ${isAi ? 'pill-gov-ai' : 'pill-gov-real'}">${esc(isAi ? t('verdictAi') : t('verdictReal'))}</span></td>
        <td class="num">${pct(v.confidence_score)}</td>
        <td>${esc(levelLabel(v))}</td>
        <td>
          <button class="btn-gov-secondary btn-small" onclick="viewBatchDetail(${idx})">${esc(t('inspect'))}</button>
        </td>
      </tr>`;
  }).join('');
}

function viewBatchDetail(index) {
  if (!currentBatchData || !currentBatchData.results[index]) return;
  currentResultData = currentBatchData.results[index];
  setBaseImage(currentBatchFiles[index]);
  switchMode('single');
  renderSingleResult(currentResultData);
  const resultsGrid = document.getElementById('resultsGrid');
  if (resultsGrid) window.scrollTo({ top: resultsGrid.offsetTop - 90, behavior: 'smooth' });
}

function updateHeatmapOpacity(val) {
  const heatmapImg = document.getElementById('heatmapImage');
  if (heatmapImg) heatmapImg.style.opacity = val / 100;
}

function switchTab(tabId, btnElement) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn-gov').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId)?.classList.add('active');
  btnElement?.classList.add('active');
}
