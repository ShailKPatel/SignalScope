// SignalScope Client Engine - Takniki Vibhag
// SIH 2026 Internal Hackathon Submission (Problem Statement C-433)

let currentResultData = null;
let currentBatchData = null;
let activeMode = 'single'; // 'single' | 'folder' | 'team' | 'about'
let currentLang = 'en'; // 'en' | 'hi'

const translations = {
  en: {
    modalTitle: "IMPORTANT PUBLIC ADVISORY / मुख्य सार्वजनिक परामर्श",
    modalTitleEng: "Media Authenticity & Misinformation Prevention Portal",
    modalDescEng: "Welcome to SignalScope. This platform analyzes submitted imagery using neural forensic classification and Grad-CAM visual anomaly localization to evaluate synthetic media likelihood.",
    modalTitleHin: "मीडिया प्रामाणिकता एवं भ्रामक सूचना रोकथाम पोर्टल",
    modalDescHin: "सिग्नलस्कोप पोर्टल में आपका स्वागत है। यह मंच सिंथेटिक मीडिया और डीपफेक छवियों की संभावना का मूल्यांकन करने के लिए फॉरेंसिक विश्लेषण प्रदान करता है।",
    btnModalClose: "PROCEED TO PORTAL / पोर्टल पर आगे बढ़ें",
    skipLink: "Skip to main content",
    govSubTitleTop: "SignalScope Media Forensics & Authenticity Engine",
    textSizeLabel: "Text Size:",
    btnContrast: "High Contrast",
    govtOfIndia: "SIGNALSCOPE FORENSIC ENGINE",
    mainPortalTitle: "SIGNAL SCOPE",
    directorate: "Media Verification & Authenticity Platform",
    emblemSub: "AUTHENTICITY ENGINE",
    leader1Name: "Shri A. Sharma",
    leader1Role: "Team Leader (Takniki Vibhag)",
    leader2Name: "Dr. P. Verma",
    leader2Role: "Principal AI Research Lead",
    menuSingle: "Single Image Submission Desk",
    menuFolder: "Directory Batch Audit Desk",
    menuTeam: "Team Directorate Members (6)",
    menuDirectives: "Technical Methodology",
    tickerLabel: "PUBLIC INTEREST",
    tickerText: "Independent media analysis desk created to counter AI-generated synthetic media, deepfakes, and fake news. Evaluation standard: Calibrated Likelihood Assessment.",
    serviceGridTitle: "Forensic Verification Desks & Services",
    tollFree: "Helpdesk: 1800-SIH-2026 (Toll Free)",
    tile1Title: "Single Artifact Inspection",
    tile1Desc: "Neural classification & visual Grad-CAM heatmap localization desk.",
    tile2Title: "Directory Batch Audit",
    tile2Desc: "High-throughput bulk image scanning for departmental compliance.",
    tile3Title: "Architecture Attribution",
    tile3Desc: "Identify Diffusion, GAN, or Transformer generator origins (Module B).",
    tile4Title: "Multimodal Claim Check",
    tile4Desc: "CLIP semantic alignment for checking image-caption claim consistency (Module E).",
    breadcrumbHome: "Home",
    breadcrumbCurrentSingle: "Single Artifact Submission Desk",
    breadcrumbCurrentFolder: "Directory Batch Audit Desk",
    breadcrumbCurrentTeam: "Team Directorate & Project Contributors (6)",
    breadcrumbCurrentAbout: "Technical Methodology & Forensic Specifications",
    cardTitleSingle: "Single Artifact Submission Desk",
    ackSingle: "Ref: SIG-2026-DESK-A",
    dropTitleSingle: "Select or Drag Image File for Forensic Testing",
    dropSubSingle: "Upload single image artifact (JPEG, PNG, WebP) to perform neural classification, Grad-CAM visual anomaly localization, generator architecture attribution, and metadata provenance evaluation.",
    btnChooseSingle: "Choose Image File",
    captionLabel: "Multimodal Claim Verification (Module E)",
    captionPlaceholder: "Enter associated text caption, claim statement, or news context to verify CLIP semantic match...",
    cardTitleFolder: "Directory Batch Audit Desk",
    ackFolder: "Ref: SIG-2026-AUDIT-B",
    dropTitleFolder: "Select Directory Folder for High-Throughput Audit",
    dropSubFolder: "Submit complete image directories for automated high-throughput media verification, batch threat statistics, and departmental compliance reporting.",
    btnChooseFolder: "Choose Image Directory",
    teamSectionTitle: "Team Directorate & Project Contributors (Takniki Vibhag)",
    teamSectionSub: "SIH 2026 • Problem Statement C-433",
    m1Name: "Shri A. Sharma",
    m1Desig: "Team Leader & System Architect",
    m1Role: "Overall project coordination, core architecture design, and SIH 2026 pipeline deployment.",
    m1Badge: "Lead Contributor",
    m2Name: "Dr. P. Verma",
    m2Desig: "Principal AI Research Lead",
    m2Role: "Neural network training, generalisation splits, and frequency domain artifact extraction.",
    m2Badge: "AI Core",
    m3Name: "Shri R. Patel",
    m3Desig: "Computer Vision & Grad-CAM Specialist",
    m3Role: "Module A Explainability engine, visual cue localization, and Layer-CAM heatmap generation.",
    m3Badge: "Module A Lead",
    m4Name: "Smt. S. Gupta",
    m4Desig: "Multimodal NLP & CLIP Alignment Lead",
    m4Role: "Module E text-image semantic matching, claim consistency verification, and prompt auditing.",
    m4Badge: "Module E Lead",
    m5Name: "Shri V. Iyer",
    m5Desig: "Security & Active Defense Specialist",
    m5Role: "Module D EXIF/C2PA metadata parser and Module G adversarial attack testing framework.",
    m5Badge: "Module D & G Lead",
    m6Name: "Shri K. Singh",
    m6Desig: "Web Infrastructure & Full-Stack Engineer",
    m6Role: "Module F Real-time web application dashboard, GIGW UI design system, and FastAPI integration.",
    m6Badge: "Module F Lead",
    aboutHeading: "Technical Methodology & Forensic Framework Specifications",
    aboutSub: "SIH 2026 C-433 Technical Spec",
    methodCoreTitle: "1. Mandatory Core Task: Unseen Generator Generalisation",
    methodCoreDesc: "SignalScope uses a transfer-learning convolutional backbone paired with spatial noise residual extraction (FFT/DCT high-frequency artifact analysis). Predictions are evaluated on a held-out test split featuring unseen generator architectures (Midjourney v6, SDXL, Flux, DALL-E 3) under calibrated ROC-AUC metrics.",
    methodModATitle: "2. Bonus Module A: Faithful Visual Explanations (Grad-CAM)",
    methodModADesc: "Provides localized visual saliency heatmaps highlighting exact pixel-level anomalies such as texture warping, irregular specular reflections, and anatomical flaws. Cites grounded natural language points without over-claiming certainty.",
    methodModBTitle: "3. Bonus Module B: Generator Family & Model Architecture Attribution",
    methodModBDesc: "Classifies generator families into Latent Diffusion, Generative Adversarial Networks (GANs), and Autoregressive Transformer architectures. Provides granular probability distributions across specific underlying models.",
    methodModCTitle: "4. Bonus Module C: Robustness Under Image Degradation Vectors",
    methodModCDesc: "Evaluates prediction resilience under severe lossy compression (JPEG Q30–Q90), downsampling, spatial noise addition, and social media platform re-encoding pipelines.",
    methodModDTitle: "5. Bonus Module D: Provenance, EXIF & Cryptographic C2PA Verification",
    methodModDDesc: "Extracts hardware camera EXIF records and validates Coalition for Content Provenance and Authenticity (C2PA) digital signatures to detect synthetic header tampering.",
    methodModETitle: "6. Bonus Module E: Multimodal Image-Text Alignment & Claim Verification",
    methodModEDesc: "Employs CLIP cross-modal semantic embeddings to evaluate alignment between submitted image content and accompanying news captions, identifying out-of-context misrepresentation.",
    methodModFTitle: "7. Bonus Module F: Real-time Web Dashboard & REST API Architecture",
    methodModFDesc: "Full-stack web application built following GIGW 3.0 government accessibility standards, backed by FastAPI asynchronous microservices for low-latency batch image scanning.",
    methodModGTitle: "8. Bonus Module G: Active Defense & Adversarial Attack Vulnerability Testing",
    methodModGDesc: "Tests classifier robustness against adversarial perturbation attacks (FGSM, PGD, spatial blurring) and applies targeted defensive smoothing to maintain detection reliability.",
    certHeader: "FORENSIC VERIFICATION REPORT",
    certSub: "Calibrated Likelihood Standards • SignalScope Certificate",
    certStamp: "EVALUATED",
    verdictSubtitle: "Probability Assessment Under Module Core",
    heatmapLabel: "Grad-CAM Heatmap Overlay",
    tabA: "Module A (Explanations)",
    tabB: "Module B (Attribution)",
    tabC: "Module C (Robustness)",
    tabD: "Module D (Metadata)",
    tabE: "Module E (Multimodal)",
    tabG: "Module G (Defense)",
    headTabA: "Headline Visual Artifact Cites & Explanations",
    headTabB: "Generator Family & Model Architecture Attribution",
    lblAttrFamily: "Predicted Generator Family",
    lblAttrModel: "Likely Architecture Model",
    lblFamProb: "Family Probability Breakdown",
    headTabC: "Robustness Under Image Degradation Vectors",
    lblRobustRating: "Overall Stability Rating",
    lblRobustJpeg: "Stable Under JPEG Q70 Compression",
    lblJpegCurve: "JPEG Compression Degradation Curve",
    headTabD: "Provenance & Cryptographic Header Inspection",
    lblExif: "EXIF Metadata Status",
    lblC2pa: "C2PA Signed Manifest Status",
    lblCamera: "Hardware Camera Record",
    lblMetaAssessment: "Integrity Assessment",
    headTabE: "Multimodal Image-Text Alignment & Claim Consistency",
    lblCaptionText: "Target Claim Text",
    lblClipScore: "CLIP Semantic Alignment Score",
    lblMultiAssessment: "Alignment Verdict",
    headTabG: "Adversarial Defense & Vulnerability Testing",
    batchResultsTitle: "Bulk Directory Audit Summary Results",
    statTotal: "Total Submitted Artifacts",
    statAi: "Probable Synthetic",
    statReal: "Probable Authentic Real",
    statLatency: "Total Processing Latency",
    thFilename: "Filename",
    thVerdict: "Verdict",
    thConfidence: "Confidence Score",
    thModel: "Attribution Architecture",
    thActions: "Actions",
    btnInspect: "Inspect Artifact",
    footerTitle: "SIGNAL SCOPE",
    footerDesc: "Independent Media Forensics & Authenticity Platform.",
    quickLinks: "",
    eventLabel: "",
    eventVal: "",
    teamLabel: "",
    teamVal: "",
    psLabel: "",
    psVal: "",
    compliance: "",
    disclaimerBox: "",
    footerCopyright: "SignalScope Media Forensics & Authenticity Engine"
  },
  hi: {
    modalTitle: "मुख्य सार्वजनिक परामर्श / IMPORTANT PUBLIC ADVISORY",
    modalTitleEng: "SIH 2026 Media Authenticity & Misinformation Prevention Portal",
    modalDescEng: "Welcome to SignalScope by Team Takniki Vibhag. This platform analyzes submitted imagery using neural forensic classification and Grad-CAM visual anomaly localization to evaluate synthetic media likelihood under SIH 2026 Problem Statement C-433.",
    modalTitleHin: "एसआईएच 2026 मीडिया प्रामाणिकता एवं भ्रामक सूचना रोकथाम पोर्टल",
    modalDescHin: "टीम तकनीकी विभाग के सिग्नलस्कोप पोर्टल में आपका स्वागत है। यह मंच एसआईएच 2026 समस्या कथन C-433 के तहत सिंथेटिक मीडिया और डीपफेक छवियों की संभावना का मूल्यांकन करने के लिए फॉरेंसिक विश्लेषण प्रदान करता है।",
    btnModalClose: "पोर्टल पर आगे बढ़ें / PROCEED TO PORTAL",
    skipLink: "मुख्य सामग्री पर जाएं",
    govSubTitleTop: "एसआईएच 2026 आंतरिक हैकाथॉन प्रविष्टि • टीम तकनीकी विभाग",
    textSizeLabel: "पाठ आकार:",
    btnContrast: "उच्च विपरीत",
    govtOfIndia: "एसआईएच 2026 फॉरेंसिक इंजन",
    mainPortalTitle: "तकनीकी विभाग",
    directorate: "तकनीकी मीडिया सत्यापन एवं भ्रामक सूचना निवारण प्रणाली",
    emblemSub: "एसआईएच 2026 C-433",
    leader1Name: "श्री ए. शर्मा",
    leader1Role: "टीम लीडर (तकनीकी विभाग)",
    leader2Name: "डॉ. पी. वर्मा",
    leader2Role: "प्रधान एआई अनुसंधान प्रमुख",
    menuSingle: "एकल छवि प्रस्तुति डेस्क",
    menuFolder: "निर्देशिका बैच लेखापरीक्षा डेस्क",
    menuTeam: "टीम निदेशालय के सदस्य (6)",
    menuDirectives: "तकनीकी कार्यप्रणाली",
    tickerLabel: "जनहित सूचना",
    tickerText: "एआई-जनरेटेड सिंथेटिक मीडिया, डीपफेक और फर्जी खबरों की रोकथाम के लिए निर्मित स्वतंत्र मीडिया विश्लेषण डेस्क। मूल्यांकन मानक: कैलिब्रेटेड संभावना आकलन।",
    serviceGridTitle: "फॉरेंसिक सत्यापन डेस्क और सेवाएं",
    tollFree: "हेल्पडेस्क: 1800-SIH-2026 (टोल फ्री)",
    tile1Title: "एकल कलाकृति निरीक्षण",
    tile1Desc: "तंत्रिका वर्गीकरण और दृश्य ग्रैड-कैम विसंगति स्थानीयकरण डेस्क।",
    tile2Title: "निर्देशिका बैच लेखापरीक्षा",
    tile2Desc: "विभागीय अनुपालन के लिए उच्च-थ्रूपुट थोक छवि स्कैनिंग।",
    tile3Title: "आर्किटेक्चर एट्रिब्यूशन",
    tile3Desc: "डिफ्यूज़न, GAN या ट्रांसफॉर्मर जनरेटर मूल की पहचान करें (मॉड्यूल B)।",
    tile4Title: "मल्टीमॉडल दावा जांच",
    tile4Desc: "छवि-कैप्शन दावे की निरंतरता की जांच के लिए CLIP सिमेंटिक संरेखण (मॉड्यूल E)।",
    breadcrumbHome: "होम",
    breadcrumbCurrentSingle: "एकल कलाकृति प्रस्तुति डेस्क",
    breadcrumbCurrentFolder: "निर्देशिका बैच लेखापरीक्षा डेस्क",
    breadcrumbCurrentTeam: "टीम निदेशालय एवं परियोजना योगदानकर्ता (6)",
    breadcrumbCurrentAbout: "तकनीकी कार्यप्रणाली एवं फॉरेंसिक विनिर्देश",
    cardTitleSingle: "एकल कलाकृति प्रस्तुति डेस्क",
    ackSingle: "संदर्भ: SIG-2026-DESK-A",
    dropTitleSingle: "फॉरेंसिक परीक्षण के लिए छवि फ़ाइल चुनें या खींचें",
    dropSubSingle: "तंत्रिका वर्गीकरण, ग्रैड-कैम दृश्य विसंगति स्थानीयकरण, जनरेटर आर्किटेक्चर एट्रिब्यूशन और मेटाडेटा उत्पत्ति मूल्यांकन करने के लिए एकल छवि कलाकृति (JPEG, PNG, WebP) अपलोड करें।",
    btnChooseSingle: "छवि फ़ाइल चुनें",
    captionLabel: "मल्टीमॉडल दावा सत्यापन (मॉड्यूल E)",
    captionPlaceholder: "CLIP सिमेंटिक मिलान की पुष्टि करने के लिए संबंधित पाठ कैप्शन, दावा कथन या समाचार संदर्भ दर्ज करें...",
    cardTitleFolder: "निर्देशिका बैच लेखापरीक्षा डेस्क",
    ackFolder: "संदर्भ: SIG-2026-AUDIT-B",
    dropTitleFolder: "उच्च-थ्रूपुट लेखापरीक्षा के लिए निर्देशिका फ़ोल्डर चुनें",
    dropSubFolder: "स्वचालित उच्च-थ्रूपुट मीडिया सत्यापन, बैच खतरे के आंकड़ों और विभागीय अनुपालन रिपोर्टिंग के लिए पूर्ण चित्र निर्देशिका जमा करें।",
    btnChooseFolder: "छवि निर्देशिका चुनें",
    teamSectionTitle: "टीम निदेशालय एवं परियोजना योगदानकर्ता (तकनीकी विभाग)",
    teamSectionSub: "एसआईएच 2026 • समस्या कथन C-433",
    m1Name: "श्री ए. शर्मा",
    m1Desig: "टीम लीडर एवं सिस्टम आर्किटेक्ट",
    m1Role: "समग्र परियोजना समन्वय, कोर आर्किटेक्चर डिज़ाइन और एसआईएच 2026 पाइपलाइन तैनाती।",
    m1Badge: "प्रमुख योगदानकर्ता",
    m2Name: "डॉ. पी. वर्मा",
    m2Desig: "प्रधान एआई अनुसंधान प्रमुख",
    m2Role: "न्यूरल नेटवर्क प्रशिक्षण, सामान्यीकरण विभाजन और आवृत्ति डोमेन कलाकृति निष्कर्षण।",
    m2Badge: "एआई कोर",
    m3Name: "श्री आर. पटेल",
    m3Desig: "कंप्यूटर विजन एवं ग्रैड-कैम विशेषज्ञ",
    m3Role: "मॉड्यूल A स्पष्टीकरण इंजन, दृश्य संकेत स्थानीयकरण और लेयर-कैम हीटमैप निर्माण।",
    m3Badge: "मॉड्यूल A प्रमुख",
    m4Name: "श्रीमती एस. गुप्ता",
    m4Desig: "मल्टीमॉडल एनएलपी एवं CLIP संरेखण प्रमुख",
    m4Role: "मॉड्यूल E पाठ-छवि सिमेंटिक मिलान, दावा निरंतरता सत्यापन और प्रॉम्प्ट लेखापरीक्षा।",
    m4Badge: "मॉड्यूल E प्रमुख",
    m5Name: "श्री वी. अय्यर",
    m5Desig: "सुरक्षा एवं सक्रिय रक्षा विशेषज्ञ",
    m5Role: "मॉड्यूल D EXIF/C2PA मेटाडेटा पार्सर और मॉड्यूल G प्रतिकूल हमला परीक्षण ढांचा।",
    m5Badge: "मॉड्यूल D एवं G प्रमुख",
    m6Name: "श्री के. सिंह",
    m6Desig: "वेब इंफ्रास्ट्रक्चर एवं फुल-स्टैक इंजीनियर",
    m6Role: "मॉड्यूल F वास्तविक समय वेब एप्लिकेशन डैशबोर्ड, GIGW UI डिज़ाइन सिस्टम और FastAPI एकीकरण।",
    m6Badge: "मॉड्यूल F प्रमुख",
    aboutHeading: "तकनीकी कार्यप्रणाली एवं फॉरेंसिक ढांचा विनिर्देश",
    aboutSub: "एसआईएच 2026 C-433 तकनीकी विवरण",
    methodCoreTitle: "1. अनिवार्य कोर कार्य: अनदेखे जनरेटर का सामान्यीकरण",
    methodCoreDesc: "सिग्नलस्कोप ट्रांसफर-लर्निंग कॉन्वोल्यूशनल बैकबोन के साथ स्थानिक शोर अवशिष्ट निष्कर्षण (FFT/DCT उच्च-आवृत्ति कलाकृति विश्लेषण) का उपयोग करता है। पूर्वानुमानों का मूल्यांकन अनदेखे जनरेटर आर्किटेक्चर (Midjourney v6, SDXL, Flux, DALL-E 3) पर कैलिब्रेटेड ROC-AUC मेट्रिक्स के तहत किया जाता है।",
    methodModATitle: "2. बोनस मॉड्यूल A: विश्वसनीय दृश्य स्पष्टीकरण (Grad-CAM)",
    methodModADesc: "सटीक पिक्सेल-स्तरीय विसंगतियों जैसे टेक्सचर वारपिंग, अनियमित स्पेक्युलर परावर्तन और शारीरिक त्रुटियों को उजागर करने वाले स्थानीयकृत दृश्य गर्मी मानचित्र (हीटमैप) प्रदान करता है।",
    methodModBTitle: "3. बोनस मॉड्यूल B: जनरेटर परिवार एवं मॉडल आर्किटेक्चर एट्रिब्यूशन",
    methodModBDesc: "जनरेटर परिवारों को लेटेंट डिफ्यूज़न, जनरेटिव एडवर्सरियल नेटवर्क (GAN) और ऑटोरेग्रेसिव ट्रांसफॉर्मर आर्किटेक्चर में वर्गीकृत करता है। विशिष्ट अंतर्निहित मॉडलों में विस्तृत संभावना वितरण प्रदान करता है।",
    methodModCTitle: "4. बोनस मॉड्यूल C: छवि क्षरण वैक्टर के तहत मजबूती",
    methodModCDesc: "गंभीर हानिकारक संपीड़न (JPEG Q30-Q90), डाउनसैंपलिंग, स्थानिक शोर जोड़ने और सोशल मीडिया री-एंकोडिंग पाइपलाइनों के तहत पूर्वानुमान स्थिरता का मूल्यांकन करता है।",
    methodModDTitle: "5. बोनस मॉड्यूल D: उत्पत्ति, EXIF एवं क्रिप्टोग्राफिक C2PA सत्यापन",
    methodModDDesc: "हार्डवेयर कैमरा EXIF रिकॉर्ड निकालता है और सिंथेटिक हेडर हेरफेर का पता लगाने के लिए सामग्री उत्पत्ति और प्रामाणिकता गठबंधन (C2PA) डिजिटल हस्ताक्षरों को सत्यापित करता है।",
    methodModETitle: "6. बोनस मॉड्यूल E: मल्टीमॉडल छवि-पाठ संरेखण एवं दावा सत्यापन",
    methodModEDesc: "जमा की गई छवि सामग्री और संलग्न समाचार कैप्शन के बीच संरेखण का मूल्यांकन करने के लिए CLIP क्रॉस-मॉडल सिमेंटिक एम्बेडिंग का उपयोग करता है, जिससे संदर्भ से बाहर गलत प्रस्तुति की पहचान होती है।",
    methodModFTitle: "7. बोनस मॉड्यूल F: वास्तविक समय वेब डैशबोर्ड एवं REST API आर्किटेक्चर",
    methodModFDesc: "GIGW 3.0 सरकारी पहुंच मानकों के अनुसार निर्मित फुल-स्टैक वेब एप्लिकेशन, जो कम-विलंबता थोक छवि स्कैनिंग के लिए FastAPI एसिंक्रोनस माइक्रोसर्विसेज द्वारा संचालित है।",
    methodModGTitle: "8. बोनस मॉड्यूल G: सक्रिय रक्षा एवं प्रतिकूल हमला भेद्यता परीक्षण",
    methodModGDesc: "प्रतिकूल गड़बड़ी हमलों (FGSM, PGD, स्थानिक धुंधलापन) के खिलाफ क्लासिफायर की मजबूती का परीक्षण करता है और पहचान विश्वसनीयता बनाए रखने के लिए लक्षित रक्षात्मक स्मूथिंग लागू करता है।",
    certHeader: "फॉरेंसिक सत्यापन रिपोर्ट",
    certSub: "कैलिब्रेटेड संभावना मानक • सिग्नलस्कोप प्रमाणपत्र",
    certStamp: "मूल्यांकित",
    verdictSubtitle: "मॉड्यूल कोर के तहत संभावना आकलन",
    heatmapLabel: "ग्रैड-कैम हीटमैप ओवरले",
    tabA: "मॉड्यूल A (स्पष्टीकरण)",
    tabB: "मॉड्यूल B (एट्रिब्यूशन)",
    tabC: "मॉड्यूल C (मजबूती)",
    tabD: "मॉड्यूल D (मेटाडेटा)",
    tabE: "मॉड्यूल E (मल्टीमॉडल)",
    tabG: "मॉड्यूल G (रक्षा)",
    headTabA: "मुख्य दृश्य कलाकृति उद्धरण एवं स्पष्टीकरण",
    headTabB: "जनरेटर परिवार एवं मॉडल आर्किटेक्चर एट्रिब्यूशन",
    lblAttrFamily: "अनुमानित जनरेटर परिवार",
    lblAttrModel: "संभावित आर्किटेक्चर मॉडल",
    lblFamProb: "परिवार संभावना विवरण",
    headTabC: "छवि क्षरण वैक्टर के तहत मजबूती",
    lblRobustRating: "समग्र स्थिरता रेटिंग",
    lblRobustJpeg: "JPEG Q70 संपीड़न के तहत स्थिर",
    lblJpegCurve: "JPEG संपीड़न क्षरण वक्र",
    headTabD: "उत्पत्ति एवं क्रिप्टोग्राफिक हेडर निरीक्षण",
    lblExif: "EXIF मेटाडेटा स्थिति",
    lblC2pa: "C2PA हस्ताक्षरित मेनिफेस्ट स्थिति",
    lblCamera: "हार्डवेयर कैमरा रिकॉर्ड",
    lblMetaAssessment: "सत्यनिष्ठा मूल्यांकन",
    headTabE: "मल्टीमॉडल छवि-पाठ संरेखण एवं दावा निरंतरता",
    lblCaptionText: "लक्ष्य दावा पाठ",
    lblClipScore: "CLIP सिमेंटिक संरेखण स्कोर",
    lblMultiAssessment: "संरेखण निर्णय",
    headTabG: "प्रति-प्रतिकूल रक्षा एवं भेद्यता परीक्षण",
    batchResultsTitle: "थोक निर्देशिका लेखापरीक्षा सारांश परिणाम",
    statTotal: "कुल जमा कलाकृतियां",
    statAi: "संभावित सिंथेटिक (एआई)",
    statReal: "संभावित प्रामाणिक (वास्तविक)",
    statLatency: "कुल प्रसंस्करण विलंबता",
    thFilename: "फ़ाइल का नाम",
    thVerdict: "निर्णय",
    thConfidence: "विश्वास स्कोर",
    thModel: "एट्रिब्यूशन आर्किटेक्चर",
    thActions: "कार्रवाई",
    btnInspect: "कलाकृति का निरीक्षण करें",
    footerTitle: "तकनीकी विभाग • TAKNIKI VIBHAG",
    footerDesc: "स्मार्ट इंडिया हैकाथॉन (एसआईएच 2026) के लिए निर्मित स्वतंत्र मीडिया फॉरेंसिक मंच। सिंथेटिक मीडिया के स्वचालित वर्गीकरण, दृश्य विसंगति स्थानीयकरण और जनरेटर एट्रिब्यूशन के लिए डिज़ाइन किया गया।",
    quickLinks: "हैकाथॉन विवरण",
    eventLabel: "कार्यक्रम:",
    eventVal: "एसआईएच 2026 आंतरिक हैकाथॉन",
    teamLabel: "टीम:",
    teamVal: "तकनीकी विभाग",
    psLabel: "समस्या कथन:",
    psVal: "C-433",
    compliance: "अस्वीकरण",
    disclaimerBox: "सिग्नलस्कोप (तकनीकी विभाग) एक अकादमिक परियोजना है। यह भारत सरकार या किसी भी मंत्रालय से संबद्ध या स्वीकृत नहीं है।",
    footerCopyright: "सिग्नलस्कोप मीडिया फॉरेंसिक सुइट • एसआईएच 2026 के लिए टीम तकनीकी विभाग द्वारा निर्मित"
  }
};

function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) {
    el.innerText = text;
  }
}

function safeSetPlaceholder(id, placeholder) {
  const el = document.getElementById(id);
  if (el) {
    el.placeholder = placeholder;
  }
}

document.addEventListener('DOMContentLoaded', () => {
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
  if (delta === -1) {
    document.body.classList.add('font-sm');
  } else if (delta === 1) {
    document.body.classList.add('font-lg');
  }
}

function toggleContrast() {
  document.body.classList.toggle('contrast-high');
}

function checkHashRoute() {
  const hash = window.location.hash.toLowerCase();
  if (hash === '#about' || hash === '#directives' || hash === '#methodology') {
    switchMode('about');
  } else if (hash === '#team') {
    switchMode('team');
  } else if (hash === '#folder' || hash === '#batch') {
    switchMode('folder');
  } else if (hash === '#single') {
    switchMode('single');
  }
}

function setLanguage(lang) {
  currentLang = lang;
  const t = translations[lang];

  const btnEn = document.getElementById('btnLangEN');
  const btnHi = document.getElementById('btnLangHI');
  if (btnEn) btnEn.classList.toggle('active', lang === 'en');
  if (btnHi) btnHi.classList.toggle('active', lang === 'hi');

  safeSetText('txtModalTitle', t.modalTitle);
  safeSetText('txtModalTitleEng', t.modalTitleEng);
  safeSetText('txtModalDescEng', t.modalDescEng);
  safeSetText('txtModalTitleHin', t.modalTitleHin);
  safeSetText('txtModalDescHin', t.modalDescHin);
  safeSetText('txtBtnModalClose', t.btnModalClose);

  safeSetText('txtSkipLink', t.skipLink);
  safeSetText('txtGovSubTitleTop', t.govSubTitleTop);
  safeSetText('txtTextSizeLabel', t.textSizeLabel);
  safeSetText('btnContrast', t.btnContrast);
  safeSetText('txtGovtOfIndia', t.govtOfIndia);
  safeSetText('txtMainPortalTitle', t.mainPortalTitle);
  safeSetText('txtDirectorate', t.directorate);
  safeSetText('txtEmblemSub', t.emblemSub);

  safeSetText('txtLeader1Name', t.leader1Name);
  safeSetText('txtLeader1Role', t.leader1Role);
  safeSetText('txtLeader2Name', t.leader2Name);
  safeSetText('txtLeader2Role', t.leader2Role);

  safeSetText('txtMenuSingle', t.menuSingle);
  safeSetText('txtMenuFolder', t.menuFolder);
  safeSetText('txtMenuTeam', t.menuTeam);
  safeSetText('txtMenuDirectives', t.menuDirectives);

  safeSetText('txtTickerLabel', t.tickerLabel);
  safeSetText('txtTickerText', t.tickerText);

  safeSetText('txtServiceGridTitle', t.serviceGridTitle);
  safeSetText('txtTollFree', t.tollFree);
  safeSetText('txtTile1Title', t.tile1Title);
  safeSetText('txtTile1Desc', t.tile1Desc);
  safeSetText('txtTile2Title', t.tile2Title);
  safeSetText('txtTile2Desc', t.tile2Desc);
  safeSetText('txtTile3Title', t.tile3Title);
  safeSetText('txtTile3Desc', t.tile3Desc);
  safeSetText('txtTile4Title', t.tile4Title);
  safeSetText('txtTile4Desc', t.tile4Desc);

  safeSetText('txtBreadcrumbHome', t.breadcrumbHome);
  updateBreadcrumbText();

  safeSetText('txtCardTitleSingle', t.cardTitleSingle);
  safeSetText('txtAckSingle', t.ackSingle);
  safeSetText('txtDropTitleSingle', t.dropTitleSingle);
  safeSetText('txtDropSubSingle', t.dropSubSingle);
  safeSetText('txtBtnChooseSingle', t.btnChooseSingle);
  safeSetText('txtCaptionLabel', t.captionLabel);
  safeSetPlaceholder('captionInput', t.captionPlaceholder);

  safeSetText('txtCardTitleFolder', t.cardTitleFolder);
  safeSetText('txtAckFolder', t.ackFolder);
  safeSetText('txtDropTitleFolder', t.dropTitleFolder);
  safeSetText('txtDropSubFolder', t.dropSubFolder);
  safeSetText('txtBtnChooseFolder', t.btnChooseFolder);

  safeSetText('txtTeamSectionTitle', t.teamSectionTitle);
  safeSetText('txtTeamSectionSub', t.teamSectionSub);
  safeSetText('txtM1Name', t.m1Name);
  safeSetText('txtM1Desig', t.m1Desig);
  safeSetText('txtM1Role', t.m1Role);
  safeSetText('txtM1Badge', t.m1Badge);

  safeSetText('txtM2Name', t.m2Name);
  safeSetText('txtM2Desig', t.m2Desig);
  safeSetText('txtM2Role', t.m2Role);
  safeSetText('txtM2Badge', t.m2Badge);

  safeSetText('txtM3Name', t.m3Name);
  safeSetText('txtM3Desig', t.m3Desig);
  safeSetText('txtM3Role', t.m3Role);
  safeSetText('txtM3Badge', t.m3Badge);

  safeSetText('txtM4Name', t.m4Name);
  safeSetText('txtM4Desig', t.m4Desig);
  safeSetText('txtM4Role', t.m4Role);
  safeSetText('txtM4Badge', t.m4Badge);

  safeSetText('txtM5Name', t.m5Name);
  safeSetText('txtM5Desig', t.m5Desig);
  safeSetText('txtM5Role', t.m5Role);
  safeSetText('txtM5Badge', t.m5Badge);

  safeSetText('txtM6Name', t.m6Name);
  safeSetText('txtM6Desig', t.m6Desig);
  safeSetText('txtM6Role', t.m6Role);
  safeSetText('txtM6Badge', t.m6Badge);

  safeSetText('txtAboutHeading', t.aboutHeading);
  safeSetText('txtAboutSub', t.aboutSub);
  safeSetText('txtMethodCoreTitle', t.methodCoreTitle);
  safeSetText('txtMethodCoreDesc', t.methodCoreDesc);
  safeSetText('txtMethodModATitle', t.methodModATitle);
  safeSetText('txtMethodModADesc', t.methodModADesc);
  safeSetText('txtMethodModBTitle', t.methodModBTitle);
  safeSetText('txtMethodModBDesc', t.methodModBDesc);
  safeSetText('txtMethodModCTitle', t.methodModCTitle);
  safeSetText('txtMethodModCDesc', t.methodModCDesc);
  safeSetText('txtMethodModDTitle', t.methodModDTitle);
  safeSetText('txtMethodModDDesc', t.methodModDDesc);
  safeSetText('txtMethodModETitle', t.methodModETitle);
  safeSetText('txtMethodModEDesc', t.methodModEDesc);
  safeSetText('txtMethodModFTitle', t.methodModFTitle);
  safeSetText('txtMethodModFDesc', t.methodModFDesc);
  safeSetText('txtMethodModGTitle', t.methodModGTitle);
  safeSetText('txtMethodModGDesc', t.methodModGDesc);

  safeSetText('txtCertHeader', t.certHeader);
  safeSetText('txtCertSub', t.certSub);
  safeSetText('txtCertStamp', t.certStamp);
  safeSetText('verdictSubtitle', t.verdictSubtitle);
  safeSetText('txtHeatmapLabel', t.heatmapLabel);

  safeSetText('txtTabA', t.tabA);
  safeSetText('txtTabB', t.tabB);
  safeSetText('txtTabC', t.tabC);
  safeSetText('txtTabD', t.tabD);
  safeSetText('txtTabE', t.tabE);
  safeSetText('txtTabG', t.tabG);

  safeSetText('txtHeadTabA', t.headTabA);
  safeSetText('txtHeadTabB', t.headTabB);
  safeSetText('lblAttrFamily', t.lblAttrFamily);
  safeSetText('lblAttrModel', t.lblAttrModel);
  safeSetText('lblFamProb', t.lblFamProb);

  safeSetText('txtHeadTabC', t.headTabC);
  safeSetText('lblRobustRating', t.lblRobustRating);
  safeSetText('lblRobustJpeg', t.lblRobustJpeg);
  safeSetText('lblJpegCurve', t.lblJpegCurve);

  safeSetText('txtHeadTabD', t.headTabD);
  safeSetText('lblExif', t.lblExif);
  safeSetText('lblC2pa', t.lblC2pa);
  safeSetText('lblCamera', t.lblCamera);
  safeSetText('lblMetaAssessment', t.lblMetaAssessment);

  safeSetText('txtHeadTabE', t.headTabE);
  safeSetText('lblCaptionText', t.lblCaptionText);
  safeSetText('lblClipScore', t.lblClipScore);
  safeSetText('lblMultiAssessment', t.lblMultiAssessment);

  safeSetText('txtHeadTabG', t.headTabG);

  safeSetText('txtBatchResultsTitle', t.batchResultsTitle);
  safeSetText('txtStatTotal', t.statTotal);
  safeSetText('txtStatAi', t.statAi);
  safeSetText('txtStatReal', t.statReal);
  safeSetText('txtStatLatency', t.statLatency);

  safeSetText('txtThFilename', t.thFilename);
  safeSetText('txtThVerdict', t.thVerdict);
  safeSetText('txtThConfidence', t.thConfidence);
  safeSetText('txtThModel', t.thModel);
  safeSetText('txtThActions', t.thActions);

  safeSetText('txtFooterTitle', t.footerTitle);
  safeSetText('txtFooterDesc', t.footerDesc);
  safeSetText('txtQuickLinks', t.quickLinks);
  safeSetText('txtEventLabel', t.eventLabel);
  safeSetText('txtEventVal', t.eventVal);
  safeSetText('txtTeamLabel', t.teamLabel);
  safeSetText('txtTeamVal', t.teamVal);
  safeSetText('txtPsLabel', t.psLabel);
  safeSetText('txtPsVal', t.psVal);
  safeSetText('txtCompliance', t.compliance);
  safeSetText('txtDisclaimerBox', t.disclaimerBox);
  safeSetText('txtFooterCopyright', t.footerCopyright);

  if (currentResultData) {
    renderSingleResult(currentResultData);
  }

  if (currentBatchData) {
    renderBatchResult(currentBatchData);
  }
}

function updateBreadcrumbText() {
  const t = translations[currentLang];
  const el = document.getElementById('txtBreadcrumbCurrent');
  if (!el) return;

  if (activeMode === 'single') {
    el.innerText = t.breadcrumbCurrentSingle;
  } else if (activeMode === 'folder') {
    el.innerText = t.breadcrumbCurrentFolder;
  } else if (activeMode === 'team') {
    el.innerText = t.breadcrumbCurrentTeam;
  } else if (activeMode === 'about') {
    el.innerText = t.breadcrumbCurrentAbout;
  }
}

function switchMode(mode) {
  activeMode = mode;
  
  const btnSingle = document.getElementById('btnMenuSingle');
  const btnFolder = document.getElementById('btnMenuFolder');
  const btnTeam = document.getElementById('btnMenuTeam');
  const btnAbout = document.getElementById('btnMenuAbout');

  const dropSingle = document.getElementById('dropZoneSingle');
  const dropFolder = document.getElementById('dropZoneFolder');
  const teamSection = document.getElementById('teamSection');
  const aboutSection = document.getElementById('aboutSection');
  const batchSection = document.getElementById('batchSection');
  const resultsGrid = document.getElementById('resultsGrid');
  
  [btnSingle, btnFolder, btnTeam, btnAbout].forEach(b => {
    if (b) b.classList.remove('active');
  });

  if (dropSingle) dropSingle.style.display = 'none';
  if (dropFolder) dropFolder.style.display = 'none';
  if (teamSection) teamSection.style.display = 'none';
  if (aboutSection) aboutSection.style.display = 'none';
  if (batchSection) batchSection.style.display = 'none';

  if (mode === 'single') {
    if (btnSingle) btnSingle.classList.add('active');
    if (dropSingle) dropSingle.style.display = 'block';
    if (currentResultData && resultsGrid) resultsGrid.style.display = 'grid';
  } else if (mode === 'folder') {
    if (btnFolder) btnFolder.classList.add('active');
    if (dropFolder) dropFolder.style.display = 'block';
    if (currentBatchData && batchSection) batchSection.style.display = 'block';
    if (resultsGrid) resultsGrid.style.display = 'none';
  } else if (mode === 'team') {
    if (btnTeam) btnTeam.classList.add('active');
    if (teamSection) teamSection.style.display = 'block';
    if (resultsGrid) resultsGrid.style.display = 'none';
  } else if (mode === 'about') {
    if (btnAbout) btnAbout.classList.add('active');
    if (aboutSection) aboutSection.style.display = 'block';
    if (resultsGrid) resultsGrid.style.display = 'none';
  }

  updateBreadcrumbText();
}

function setupDragAndDrop() {
  const setupZone = (dropZone, isFolder) => {
    if (!dropZone) return;
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
      }, false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        if (isFolder || files.length > 1) {
          switchMode('folder');
          processFolderFiles(files);
        } else {
          switchMode('single');
          processSingleFile(files[0]);
        }
      }
    });
  };

  setupZone(document.getElementById('dropZoneSingle'), false);
  setupZone(document.getElementById('dropZoneFolder'), true);
}

function handleSingleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    processSingleFile(files[0]);
  }
}

function handleFolderSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    processFolderFiles(files);
  }
}

async function processSingleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert(currentLang === 'hi' ? 'कृपया एक वैध छवि फ़ाइल (JPEG, PNG, WebP) चुनें।' : 'Please select a valid image file (JPEG, PNG, WebP).');
    return;
  }

  const captionInput = document.getElementById('captionInput');
  const captionText = captionInput ? captionInput.value : '';
  await uploadSingleFile(file, captionText);
}

async function processFolderFiles(fileList) {
  const files = Array.from(fileList).filter(f => f.type.startsWith('image/'));
  if (files.length === 0) {
    alert(currentLang === 'hi' ? 'चयनित निर्देशिका में कोई वैध छवि फ़ाइलें नहीं मिलीं।' : 'No valid image files found in selected directory.');
    return;
  }

  await uploadBatchFiles(files);
}

async function uploadSingleFile(file, caption) {
  const formData = new FormData();
  formData.append('file', file);
  if (caption) {
    formData.append('caption', caption);
  }

  try {
    const reader = new FileReader();
    reader.onload = (e) => {
      const baseImg = document.getElementById('baseImage');
      if (baseImg) baseImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    const resp = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    
    if (!resp.ok) throw new Error('Prediction API failed');

    const data = await resp.json();
    currentResultData = data;
    renderSingleResult(data);
  } catch (err) {
    console.error('Error analyzing image:', err);
    alert('Error connecting to SignalScope backend service: ' + err.message);
  }
}

async function uploadBatchFiles(files) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  try {
    const resp = await fetch('/api/predict-batch', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) throw new Error('Batch API failed');

    const batchData = await resp.json();
    currentBatchData = batchData;
    renderBatchResult(batchData);

    if (batchData.results && batchData.results.length > 0) {
      currentResultData = batchData.results[0];
      const firstFile = files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        const baseImg = document.getElementById('baseImage');
        if (baseImg) baseImg.src = e.target.result;
      };
      reader.readAsDataURL(firstFile);
      renderSingleResult(currentResultData);
    }
  } catch (err) {
    console.error('Error processing batch:', err);
    alert('Error processing batch upload: ' + err.message);
  }
}

function renderSingleResult(data) {
  const resultsGrid = document.getElementById('resultsGrid');
  if (resultsGrid) resultsGrid.style.display = 'grid';

  const verdict = data.verdict;
  const modules = data.modules;

  const banner = document.getElementById('verdictBanner');
  const label = document.getElementById('verdictLabel');
  const score = document.getElementById('verdictScore');

  if (label) {
    if (currentLang === 'hi') {
      label.innerText = verdict.is_ai_generated ? "संभावित नकली (एआई)" : "संभावित वास्तविक";
    } else {
      label.innerText = verdict.is_ai_generated ? "LIKELY FAKE" : "LIKELY REAL";
    }
  }

  if (score) score.innerText = `${(verdict.confidence_score * 100).toFixed(1)}%`;
  safeSetText('imageResolution', data.resolution);

  if (banner) {
    banner.className = verdict.is_ai_generated ? 'verdict-banner-gov ai' : 'verdict-banner-gov real';
  }

  const modA = modules.module_a_explainability;
  const heatmapImg = document.getElementById('heatmapImage');
  if (heatmapImg) heatmapImg.src = modA.heatmap_base64;
  safeSetText('explanationSummary', modA.summary_text);

  const cuesContainer = document.getElementById('visualCuesContainer');
  if (cuesContainer) {
    cuesContainer.innerHTML = modA.cues.map(c => `
      <div class="cue-item-gov">
        <div class="cue-header-gov">
          <span>${c.type}</span>
          <span style="color: var(--gov-navy); font-family: var(--font-mono);">${currentLang === 'hi' ? 'विश्वास' : 'Confidence'}: ${Math.round(c.confidence * 100)}%</span>
        </div>
        <div class="cue-desc-gov">${c.detail}</div>
      </div>
    `).join('');
  }

  const modB = modules.module_b_attribution;
  safeSetText('attrFamily', modB.family);
  safeSetText('attrModel', modB.specific_model);
  
  const famContainer = document.getElementById('familyProbContainer');
  if (famContainer) {
    famContainer.innerHTML = Object.entries(modB.family_probabilities).map(([fam, prob]) => `
      <div class="metric-row-gov">
        <span class="metric-label-gov">${fam}</span>
        <span class="metric-val-gov">${(prob * 100).toFixed(1)}%</span>
      </div>
    `).join('');
  }

  const modC = modules.module_c_robustness;
  safeSetText('robustnessRating', modC.overall_stability_rating);
  safeSetText('robustnessJpeg', modC.verdict_preserved_under_jpeg70 ? (currentLang === 'hi' ? "स्थिर" : "Stable") : (currentLang === 'hi' ? "क्षीण" : "Degraded"));

  const jpegContainer = document.getElementById('jpegCurveContainer');
  if (jpegContainer) {
    jpegContainer.innerHTML = modC.jpeg_degradation_curve.map(row => `
      <div class="metric-row-gov">
        <span class="metric-label-gov">${currentLang === 'hi' ? 'गुणवत्ता' : 'Quality'} ${row.quality}</span>
        <span class="metric-val-gov">${(row.confidence_retained * 100).toFixed(1)}%</span>
      </div>
    `).join('');
  }

  const modD = modules.module_d_metadata;
  safeSetText('metaHasExif', modD.has_exif ? (currentLang === 'hi' ? "उपलब्ध" : "Present") : (currentLang === 'hi' ? "हटाया गया" : "Stripped"));
  safeSetText('metaHasC2pa', modD.has_c2pa ? (currentLang === 'hi' ? "हस्ताक्षरित" : "Signed") : (currentLang === 'hi' ? "अहस्ताक्षरित" : "Unsigned"));
  safeSetText('metaCamera', modD.camera_model ? `${modD.camera_make} ${modD.camera_model}` : (currentLang === 'hi' ? "अज्ञात / लागू नहीं" : "Unknown / N/A"));
  safeSetText('metaAssessment', modD.assessment);

  const modE = modules.module_e_multimodal;
  safeSetText('multiCaption', modE.caption_text || (currentLang === 'hi' ? "कोई नहीं दिया गया" : "None provided"));
  safeSetText('multiScore', modE.semantic_similarity ? `${(modE.semantic_similarity * 100).toFixed(1)}%` : "N/A");
  safeSetText('multiAssessment', modE.assessment || (currentLang === 'hi' ? "छोड़ा गया" : "Skipped"));

  const modG = modules.module_g_active_defense;
  const defenseContainer = document.getElementById('activeDefenseContainer');
  if (defenseContainer) {
    defenseContainer.innerHTML = modG.defense_evaluations.map(ev => `
      <div class="cue-item-gov">
        <div class="cue-header-gov">
          <span>${ev.attack_type}</span>
          <span style="color: var(--gov-navy); font-family: var(--font-mono);">${currentLang === 'hi' ? 'संरक्षित स्कोर' : 'Retained'}: ${ev.accuracy_retained}</span>
        </div>
        <div class="cue-desc-gov"><strong>${currentLang === 'hi' ? 'निवारण' : 'Mitigation'}:</strong> ${ev.mitigation}</div>
      </div>
    `).join('');
  }
}

function renderBatchResult(batchData) {
  const batchSection = document.getElementById('batchSection');
  if (batchSection) batchSection.style.display = 'block';

  const t = translations[currentLang];
  const summary = batchData.summary;
  
  safeSetText('statTotalImages', batchData.total_images);
  safeSetText('statAiCount', summary.likely_ai_generated_count);
  safeSetText('statRealCount', summary.likely_real_count);
  safeSetText('statLatency', `${batchData.total_time_seconds}s`);

  const tbody = document.getElementById('batchTableBody');
  if (tbody) {
    tbody.innerHTML = batchData.results.map((item, idx) => {
      const isAi = item.verdict.is_ai_generated;
      const verdictLabel = currentLang === 'hi' ? (isAi ? 'संभावित नकली' : 'संभावित वास्तविक') : (isAi ? 'LIKELY FAKE' : 'LIKELY REAL');
      return `
        <tr>
          <td>${item.filename}</td>
          <td><span class="${isAi ? 'pill-gov pill-gov-ai' : 'pill-gov pill-gov-real'}">${verdictLabel}</span></td>
          <td>${(item.verdict.confidence_score * 100).toFixed(1)}%</td>
          <td>${item.modules.module_b_attribution.specific_model}</td>
          <td>
            <button class="btn-gov-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.75rem;" onclick="viewBatchDetail(${idx})">
              ${t.btnInspect}
            </button>
          </td>
        </tr>
      `;
    }).join('');
  }
}

function viewBatchDetail(index) {
  if (currentBatchData && currentBatchData.results[index]) {
    currentResultData = currentBatchData.results[index];
    switchMode('single');
    renderSingleResult(currentResultData);
    const resultsGrid = document.getElementById('resultsGrid');
    if (resultsGrid) {
      window.scrollTo({ top: resultsGrid.offsetTop - 90, behavior: 'smooth' });
    }
  }
}

function updateHeatmapOpacity(val) {
  const heatmapImg = document.getElementById('heatmapImage');
  if (heatmapImg) heatmapImg.style.opacity = val / 100;
}

function switchTab(tabId, btnElement) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn-gov').forEach(el => el.classList.remove('active'));

  const targetTab = document.getElementById(tabId);
  if (targetTab) targetTab.classList.add('active');
  if (btnElement) btnElement.classList.add('active');
}
