# Pair outcomes after language removal (raw, LaBSE, k=64, 311 test pairs)

Outcome counts (raw): {'blind': 201, 'separated': 83, 'reversed': 27}
Raw -> modern:

| out_raw   |   blind |   reversed |   separated |
|:----------|--------:|-----------:|------------:|
| blind     |     175 |          8 |          18 |
| reversed  |       5 |         22 |           0 |
| separated |      27 |          0 |          56 |

## Feature means by outcome

| out_raw   |   sim |   margin |   n_shared |   oov_xh |   digits |   caps_xh |   len_ratio |   lang_xh |   lang_en |   page |    year |
|:----------|------:|---------:|-----------:|---------:|---------:|----------:|------------:|----------:|----------:|-------:|--------:|
| blind     | 0.623 |    0.106 |      1.841 |    0.303 |    0.011 |     0.251 |       1.004 |     0.475 |     0.456 |  2.244 | 1887.86 |
| reversed  | 0.644 |    0.1   |      2.37  |    0.31  |    0.003 |     0.223 |       1     |     0.474 |     0.463 |  2.444 | 1887.56 |
| separated | 0.611 |    0.108 |      1.771 |    0.31  |    0.011 |     0.245 |       1     |     0.459 |     0.458 |  2.253 | 1888.67 |

## Spearman correlation with spread (p_xh of isiXhosa half minus p_xh of English half)

- sim: rho=-0.100 (p=0.0784)
- margin: rho=+0.030 (p=0.596)
- n_shared: rho=-0.035 (p=0.534)
- oov_xh: rho=+0.027 (p=0.633)
- digits: rho=+0.091 (p=0.107)
- caps_xh: rho=+0.017 (p=0.77)
- len_ratio: rho=-0.008 (p=0.882)
- lang_xh: rho=-0.170 (p=0.00271)
- lang_en: rho=-0.072 (p=0.206)
- page: rho=-0.031 (p=0.59)
- year: rho=+0.074 (p=0.192)

Outcome by sim quartile:

| sim            |   blind |   reversed |   separated |
|:---------------|--------:|-----------:|------------:|
| (0.55, 0.574]  |    0.62 |       0.06 |        0.32 |
| (0.574, 0.602] |    0.6  |       0.05 |        0.35 |
| (0.602, 0.658] |    0.7  |       0.1  |        0.19 |
| (0.658, 0.792] |    0.67 |       0.13 |        0.21 |

Outcome by page of isiXhosa half:

|   page |   blind |   reversed |   separated |
|-------:|--------:|-----------:|------------:|
|      1 |       1 |          0 |           1 |
|      2 |     153 |         15 |          60 |
|      3 |      44 |         12 |          22 |
|      4 |       3 |          0 |           0 |

## Separated most strongly

### 1891-05-21_p003_xh_024 / 1891-05-21_p003_en_039  sim=0.626 anchors=3 (bloemfontein bruce knight)  p_xh: xh=0.89 en=0.09  modern: separated  oov=0.28

- xh: sibe netamsanqa lokuqubisana nesinye sezihandiba zase ma Ngesini — inene ngenene. Siteta u Bishop wase Bloemfontein, u Right Rev. Dr. Knight. Bruce. Kube ngummangaliso ukufumana ukuba ukolisile ukusazi ngegama, ezite nezincoko ngelituba zazezinezifundiso ezendeleyo, ngapaya kokuba zibe zeziyolileyo. U Bishop “ uqonde ” kweli lisandukuvulwa elise Zimbezi apo aya kuqala umsebenzi omtsha kwabentlanga
- en: of the big success which the faith he has in the undertaking is sure to bring about. The See of Bloemfontein is far too small for such an adventerous soul as that of Dr. Knight-Bruce; and inasmuch as the recent Provincial Synod decided that there should be a new Bishopric of of Mashonaland, it would be no matter of surprise

### 1891-04-30_p002_xh_027 / 1891-04-30_p003_en_020  sim=0.566 anchors=3 (capt conner ngqamakwe)  p_xh: xh=0.90 en=0.12  modern: separated  oov=0.40

- xh: umsebenzi wobhalo zimantyi zapesheya kwe Nciba, nokunikwa kwawo abelungu bevenkile; ngapandle kwemantyi enye yase Ngqamakwe, u Capt. J. T. O’Conner ote yena akaketa bala, kodwa ete wanyula amadodana akowetu afundisiweyo alishumi elinamnye oknbhala kolubhalo beluko. Uquba ati, kungummangaliso ukuti ezimantyi zitanda kangaka ukongamela imivulo yetyal.ke, nokuyala amawetu ukuba agcine imfundo, kanti 
- en: by the Government as enumerators during the last Census. We learn from a writer in the columns of the Cape Argus, that only one Magistrate, out of six, has employed a fair number of Native enumerators in his district, and that official is Capt. J. T. O’Conner, of Ngqamakwe. The Native correspondent of the Argus says: “How absurd after this

### 1887-01-26_p003_xh_034 / 1887-01-26_p004_en_000  sim=0.750 anchors=0 ()  p_xh: xh=0.90 en=0.12  modern: separated  oov=0.22

- xh: ngako kwi Almanak zezi kolo. III. Ititshala kufuneka zisibambe isikolo ngokusondeleneyo kakulu nexesha lokususela ku 9 wakusasa ukusa ku 1 emini. Izikolo zokutungisa zingene malanga. IV. Ititshala kufuneka ukubalungiselela kwazo abantwana zibati mbo emigaqweni kanye abaviwa kuyo ngu Mhloli. V. Incwadi zamagama mazilungiswe ekupeleni kweveki zonke, nenani labatungayo malipaulwe. VI. Ukuba kuko impa
- en: manack, 3. Teachers are required to hold school as nearly as possible from 9 o’clock a.m. to 1 o’clock p.m., and the classes for sewing are to be held in the afternoon. -1. Teachers are required to prepare the children for inspection and examination in strict accordance with the published table of standards. 5. The school registers must be made

### 1893-08-09_p002_xh_053 / 1893-08-09_p003_en_021  sim=0.578 anchors=0 ()  p_xh: xh=0.89 en=0.15  modern: separated  oov=0.36

- xh: ukuwucukumisa umteto omi ngawo umbuso aziswelekile indiwo ezitnandla ezingalungiswayo kunale. (Hear, hear.) Akuko nangxokozelo ecela lomteto, kunjalo nje akuko sizatu sakunxuduzela, pantsi kwe zimeko akanakuwuxhasa lomteto. U MR. FAURE wenze inqaku lokuba abantu abangeikiva imiteto yooalulo Iwentlanga ngabo kanye abazingenisayo. Ukutsho ke 'lungu elibekekileyo lase Victoria East (Mr. Tamplin) ling
- en: grim policy of Native disarmament to execute which that proconsul secured Mr. (now Sir GORDON) SPRIGG. Again it must be remembered that the only monument, of an avowedly benevolent piece of legislation on the Statute Books of the Colony — the Native Voters’ Relief Act — was the result of Mr. HOFMEYR’S handy-work at a time our people were being


## Separated despite highest similarity

### 1886-07-28_p002_xh_032 / 1886-07-28_p003_en_018  sim=0.768 anchors=6 (augustine canterbury college hani jacob maneli)  p_xh: xh=0.58 en=0.41  modern: separated  oov=0.18

- xh: wokubhala emapepeni elozwe ngezinto zemzi abasuka kuyo. Kwelinye ipepa sifumana lenteto enge St. Augustine’s College Canterbury apo kufunda kona amadodana amabini antsundu u Mr. Jacob Maneli no Mr. Ebenezer J. Hani abasuka Emtata. Uti lowo mbhali — Olunye kusuku nezimbini endandikwesi sixeko sidala i Canterbury ndaka ndatyelelakulo Koleji inkulu yokuqeqesha abafundisi base Koloni, kwaba yinto eyol
- en: from the Colonies. Some of them are doing good service in giving their experiences in the newspapers of the town from which they hail. From one of these communications we cull the following about St. Augustine’s College, Canterbury, where two aboriginal Natives of South Africa — Mr. Jacob Maneli, and Mr. E. J. Hani, from Umtata — are studying: “

### 1886-07-28_p002_xh_023 / 1886-07-28_p003_en_004  sim=0.763 anchors=7 (john ludidi makaula nota river rode september)  p_xh: xh=0.89 en=0.27  modern: separated  oov=0.27

- xh: kuvuna, siyashuba ukuvuna siza kubakapa ukubesa apo e Rode, silindeke u Nota ukuza kwake kunye nabantu bake o Makaula no Ludidi abantu bake. Baku nyuswa ngo September ; kodwake asiyi kubani, siya Emampondweni e Rode, kanjalo siyawufuna umhlaba wama Xesibe ne St. John’s River. Makutetwe kulungiswe kuhle ngendaba zoxolo, kuba nalendlela kutiwa bayipiwe ngu Nqwiliso kodwa umhlaba ndawugqibela ungoka 
- en: Nota and his people to go over to Makaula and Ludidi. His people will be cleared in September. We don’t wish to molest anyone, we are merely marching over Pondo territory, which is the Rode. Again, we still claim Xesibeland and the St. John’s River. Some amicable arrangement must be come to. It is not impossible to arrive at a

### 1886-06-16_p002_xh_023 / 1886-06-16_p003_en_001  sim=0.763 anchors=4 (1867 1884 22 july)  p_xh: xh=0.79 en=0.38  modern: blind  oov=0.30

- xh: lomteto we Pasi basuke bawu nyatele nge nyawo becimile nganamabom. Ngenyaniso makube kuko i-ehe ngasele kwalento. Nasi eso saziso: — Isaziso se 4, 1884.] I Ofisi yo Mpatiswa.Bantsundu, E Kapa, 1 July, 1884. Ukutolikwa kwo Mteto we Pasi. Nkosi, — Njengokuba kubonakala nje ukuba amagwibi Omteto wa 22 womnyaka we 1867 akahlale egciniwe ezingqondweni okanye akaziwa kakuhle, ndiyalezwa ukuba mandalate
- en: “ something rotten in the state of Denmark ” : — Circular No. 4, 1884.] Office of the Secretary for Native Affairs, Cape Town, 1st July, 1884. INTERPRETATION OF THE PASS LAW. SIR, — As it appears that the provisions of Act No. 22 of 1867 are not always borne in mind or clearly understood, I am directed to point

### 1887-01-26_p003_xh_034 / 1887-01-26_p004_en_000  sim=0.750 anchors=0 ()  p_xh: xh=0.90 en=0.12  modern: separated  oov=0.22

- xh: ngako kwi Almanak zezi kolo. III. Ititshala kufuneka zisibambe isikolo ngokusondeleneyo kakulu nexesha lokususela ku 9 wakusasa ukusa ku 1 emini. Izikolo zokutungisa zingene malanga. IV. Ititshala kufuneka ukubalungiselela kwazo abantwana zibati mbo emigaqweni kanye abaviwa kuyo ngu Mhloli. V. Incwadi zamagama mazilungiswe ekupeleni kweveki zonke, nenani labatungayo malipaulwe. VI. Ukuba kuko impa
- en: manack, 3. Teachers are required to hold school as nearly as possible from 9 o’clock a.m. to 1 o’clock p.m., and the classes for sewing are to be held in the afternoon. -1. Teachers are required to prepare the children for inspection and examination in strict accordance with the published table of standards. 5. The school registers must be made


## Blind, highest similarity

### 1885-03-02_p003_xh_007 / 1885-03-02_p002_en_009  sim=0.792 anchors=5 (1879 1881 india lord ripon)  p_xh: xh=0.07 en=0.05  modern: blind  oov=0.24

- xh: ngo 1879, waza ngo 1881 wamnkela indawo yokuba abe pantsi kuka Lord Ripon owayesuka eIngilane esiya kuba ngu Mongame-li wayo yonke i-India. ke apo ukuba u; ekubeni ati kwaoko ayihlambulule impato yase India kububi ebebusebude baqokelelana iminyaka, ababecinga ukuba abanye, ababengenabo ubugcisa awayenabo yena, bububi obuyakupeliswa ngokutatyatwa ixesha nangonyamezelo, wapumake kuleyo indawo kwabes
- en: the proud and strong. He returned from the Soudan in 1879 and in 1881 he accepted an appointment as Private Secretary or virtual Prime Minister, under Lord RIPON who was leaving England as Viceroy of India. Finding that he would be considerably fettered in speedily cleansing the India administration of much of the debris that had accumulated for years, and

### 1887-01-26_p003_xh_035 / 1887-01-26_p004_en_002  sim=0.785 anchors=0 ()  p_xh: xh=0.05 en=0.01  modern: blind  oov=0.25

- xh: njalo-njalo mayizixele kamsinya ititshala ku Mfundisi owongamele esosikolo. VII. Nokuba yiyipina Ititshala eseyine minyaka emihlanu nangapezulu ikulomsebenzi wokufundisa iyacelwa ukuba lonto iboyixela ku Mhloli xa afikileyo eze kuhlola isikolo sayo. VIII. Ngumsebenzi wezibonda zonke zezikolo ukuqonda ukuba imali yetitshala iyabhatalwa, nokubika ku Mhloli nokuba ku Mfundisi owongamele esosikolo xa 
- en: for five years or more, is requested to mention the fact to the deputy-inspector when he makes his inspection. 8. It is the duty of all headmen schools to see that the local contributions are duly paid to the teachers, and to report to the deputyinspector or the missionary-superintendent of the school any neglect of duty (such as being absent

### 1886-09-29_p002_xh_018 / 1886-09-29_p003_en_019  sim=0.774 anchors=3 (dillwyn edward stanhope)  p_xh: xh=0.16 en=0.09  modern: blind  oov=0.13

- xh: u Mr. L. L. Dillwyn olilungu le Komiti yo Manyano Lokukusela Abantsundu, ufake umbuzo ngemicimbi yama Mpondo. Impendulo ka Mr. Edward Stanhope, Umpatiswa Mihlaba ezi Koloni ye Nkosazana ayinanto ikanyisa yona, kusuke kwavela into esasiyixela sisiti leya ntlanganiso yama Ngesi ase Kokstad nama Mpondo ayisokunceda nto ukuba itsibisile ku Sir Hercules Robinson. Incwadi zokugqibela ziya kuwavula amehl
- en: was asked in the Imperial Parliament by Mr. L. L. Dillwyn, who is a committee member of the Aborigines’ Protection Society, on Pondo matters. The reply of Mr. Edward Stanhope, the Secretary of State for the Colonies, does not change the situation by one whit, and it perfectly establishes our contention in a recent article that it is vain for

### 1887-01-26_p003_xh_007 / 1887-01-26_p004_en_013  sim=0.768 anchors=3 (bradfield glen grey)  p_xh: xh=0.64 en=0.90  modern: blind  oov=0.30

- xh: wake. Kute kusakungena u Sir Thos. Scanlen ekulauleni guququ waxhasa yena. Ngesi sipitipiti sokubiwa komhlaba waba Tembu e Glen Grey u Mr. Bradfield wawalwela kakulu amawetu, ngokoke nakuba singenakuzixela ukomelela kwazo izimvo zika Mr. Bradfield, nguyena tina sikolwa nguye kunamanye amanene la akankanyiweyo kuba wona ngo andikatali ngamalungelo abantu abamnyama; simtanda nangapezu ko Mr. Munnik 
- en: the support of Mr. Bradfield. In connection with the execrable robbery of the natives’ lands at Glen Grey he espoused the cause of our people. Therefore, while we cannot say what amount of backbone may be decerned in Mr Bradfield’s principles, we certainly should prefer him to the other gentlemen, who are either callous and careless as to native interests;


## Blind, most shared anchors

### 1886-06-16_p002_xh_033 / 1886-06-16_p003_en_022  sim=0.756 anchors=9 (22 33 innes leonard sauer sprigg tudhope upington)  p_xh: xh=0.50 en=0.89  modern: blind  oov=0.28

- xh: £25 ukuze ndibe ngonokunyula amalungu e Palamente. Bafane basityhoba nje ngalento. Bayilwile kakulu lento o Mr. Leonard, Mr. Sauer, Mr. Vintcent, Mr. Innes nabanye akwanceda nto, kwati kusaku votwa boyiswa, bona banga 22 intshaba za 33. Abona baxasi naba teteleli balomteto ukohlakele kangaka ngo Mr. Upington, Mr. Sprigg, Mr. Tudhope. no Mr. De Wet lento bayenza besekelwe ngama Bhulu angamalungu
- en: can vote. This is needless insult. Strenuously have Mr. Leonard, Mr. Sauer, Mr. Vintcent, Mr. Innes and others fought against this. In vain; they were defeated by 33 to 22. The chief supporters of this novel and outrageous proposition are Mr. Upington, Mr. Sprigg, Mr. Tudhope and Mr. de Wet, backed up by all the Dutch members in Parliament. The

### 1890-06-26_p003_xh_005 / 1890-06-26_p003_en_016  sim=0.748 anchors=9 (fuller hofmeyr innes laing merriman schermbrucker sivewright upington)  p_xh: xh=0.47 en=0.26  modern: blind  oov=0.32

- xh: becebo labapatiswa ziziteti zalonkundla ezinje ngo Mr. Sauer, Mr. Merriman, Mr. Innes, Mr. Fuller; abatelelwe ngo Mr. Hofmeyr, Mr. Sivewright, Mr. C. T. Jones nabanye. Ukuxhasa icebo labapatiswa kubaluleke o Sir T. Upington, Mr. Vintcent, Mr. Laing no Mr. Schermbrucker, namanye amadoda, asikono singapantsi kwesalawo. Ngu Sir Gordon ongekateti kuba ebanjwe yingqele le ikoyo; ngelituba ide ingxoxo y
- en: Merriman, Mr. Innes, Mr. Sivewright, Mr. Fuller. Mr. Hofmeyr, and others. In favour of the measure able speeches have been made by Mr. Vintcent, Mr. Laing, and Captain Brabant. Of Ministers, speeches have been made by Mr. Schermbrucker and Sir T. Upington. The debate stands adjourned until tomorrow, when the Premier, who is unfortunately ill, will close the Government case.

### 1887-01-26_p003_xh_039 / 1887-01-26_p004_en_005  sim=0.670 anchors=8 (april conference john kakaza mazamisa november pamla philip)  p_xh: xh=0.78 en=0.76  modern: blind  oov=0.20

- xh: Bhayi. Isimo somsebenzi kutiwa siyatembisa. Kunyulwe o Rev. T. A. Chalker no Rev. R. Matterson ukuya kwi Confrence eyakuba se Dayimani ngo April ozayo. Kweyabantsundu intlanganiso eyayingo November e Rini kwanyulwa u Rev. G. Kakaza no Mr. Philip J. Mavavana abayela iramente yabantsundu. Eyase Komani intlanganiso yabantsundu inyule u Revt C. Pamla, no Mr. John F. Mazamisa ukuya kwi Conference.
- en: in April at Kimberley. At the Native Committee held in November last at Graham’s Town the Rev. G. Kakaza and Mr. Philip Job were chosen as representatives of the natives at Conference. The Queen’s Town District Native Committee has elected the Rev. C. Pamla and Mr. John F. Mazamisa to Conference. The meeting of the Clarkebury District Committee was held

### 1893-02-02_p003_xh_015 / 1893-02-02_p003_en_034  sim=0.688 anchors=7 (cala indwana jojweni lupapasi qiba seplan sifonondile)  p_xh: xh=0.91 en=1.00  modern: blind  oov=0.30

- xh: yako. Ngokubona ubukulu besithili senze i Bhodi ezingengapantsi kwesihlanu, silinge ukuba singabi nebhodi ezininzi, ezi ke ziya kugubungela umhlaba obanzi omiweyo, nakuba abanye babefuna i Bhodi ezingapezulu. Bendingazimisa ngoluhlobo luJandelayo(1) Indwana kunye ne Seplan ne Jojweni; (2) I Cala elipezulu kunye no Lupapasi, Mcewula ne Sifonondile; (3) I Cala elisezantsi kunye ne Nqanqadu, Qiba no 
- en: although others were for more Boards. I would arrange them as follows: (1) Indwana, including Seplan and Jojweni ; (2) Upper Cala, including Lupapasi, Meewula, Sifonondile; (3) Lower Cala, including Nqanqundu, Qiba, Manzandaka ; (4) Munxe, including Tsengiwe, Upper Lufuta; (5) Lower Tsomo, including Lower Lufuta, Myalase, Hota. In regard to the matter of Title Deeds, the people cannot say


## Reversed

### 1885-03-23_p002_xh_038 / 1885-03-23_p004_en_007  sim=0.787 anchors=3 (australia french russia)  p_xh: xh=0.36 en=0.57  modern: blind  oov=0.25

- xh: yona inyaniso yokokuba isizwe sama Ngesi sinamandla kwaye kananjalo kungabonakali okokuba kungabako esinye isizwe esingafumana singxamele ukulwa naso, asiyiyo netandatyuzwayo. Ne Russia le nangoku kungokuba icinga okokuba ngokuba into engayifumanayo e-Ingilane ekunzima ukuyifumana ngamanye amaxesha, ngokukangelake kwezizipitipiti zikoyo apa e-Afrika nalentswela mvisiswano nama Jelimeni nama French
- en: English Nation is still powerful, and that it is not likely that any power would risk a war with her. Even now it is because Russia thinks she can get out of England what she thinks she cannot get at other times with the troubles in Africa, and misunderstandings with the Germans and French. The great colonies of Australia and

### 1886-06-16_p002_xh_029 / 1886-06-16_p003_en_008  sim=0.771 anchors=3 (ayliff william wood)  p_xh: xh=0.27 en=0.75  modern: reversed  oov=0.27

- xh: ku nyulwe ikati kutiwe mayenze imiteto yamalungelo nentlalo ye mpuku. I Palamente ke inyule u Mr. J. G. Wood no Mr. William Ayliff ukuba bawuqulunqe ze bawungenise ebandla lowo mteto ongasoze ubenamsebenzi emhlabeni apa, kupela into yawo kukukataza nokuqumbisa kwa noku gonyamela abantu abamnyama beli lizwe. Kodwa ke ingaba yintoni eyiyimbi owenzelwa yona, ngati akuko bani waziyo. Bapantse ukuyicas
- en: interests and convenience of the Natives on this oppressive and vexatious law. Well — Mr. J. G. WOOD and Mr. WILLIAM AYLIFF are to bring in the Bill, which will serve no earthly purpose but that of irritating and galling the black people of this country. For what else this law is wanted no one seems to know. Almost all

### 1893-02-02_p003_xh_011 / 1893-02-02_p003_en_028  sim=0.759 anchors=4 (august cala griffiths xalanga)  p_xh: xh=0.34 en=0.59  modern: reversed  oov=0.33

- xh: Figilan wase Cala, ekubeni ebhalele u Col. Griffiths, M.L. A., ngomhla we 11 August ogqitileyo, ngokumalunga nabantsundu base Xalanga kuba bemangala Izatifikiti zokumiwa komhlaba, ezinikelwa kubo ngoku kumhlaba abawunikelwa yi Komishoni yelaba Tembu, ndiyalelwe ngu Mr. Merriman ukuba mandicele ukuba makwaziswe ngokubanzi ukuba ekubeni Isatifikiti siqiniselwe kosifunayo onke amalungelo anokufunyanw
- en: Occupation of Allotments Assigned to Loyal Natives in Xalanga District. SIR, — Mr. P. Figlan of Cala, having addressed Col. Griffiths, M.L.A., under date the 15th August last representing the Xalanga Natives as objecting to “Certificates of Occupation” now being issued to them for land assigned to them by the Tembuland Commission, I am directed by Mr. Merrimen to request

### 1885-03-23_p002_xh_018 / 1885-03-23_p003_en_014  sim=0.724 anchors=3 (cala mankayi rengqe)  p_xh: xh=0.16 en=0.51  modern: reversed  oov=0.40

- xh: nguwo ngoku omayo. Oku sikuve kuqala kwincwadi yelinye lamafama akuteleyo lase Batenjini esiyishicilele ngeveki engapaya kwedluleyo. U-Mr. MANKAYI RENGQE ogama laziwayo yinkoliso yabafundi betu, usibalele esibalisela ngentlanganisq abebenayo e-Cala, abahlangene kuyo zwini linye lokukuba kucelwe umantyi ukuba abalele ku Rulumeni ukuba lomda ungavunyelwa. Umbaleli omhlope we Cape Mecury, elinyelamap
- en: an energetic native farmer which we published in our columns a fortnight ago. Mr. MANKAYI RENGQE, whose name is not new to most of our readers, sent us a short account of a meeting of native farmers at Cala, at which they resolved to ask their magistrate to write to Government not to sanction the VAN RENSBURG line. A European


## Separated in raw, blind after modernising

### 1886-06-16_p002_xh_023 / 1886-06-16_p003_en_001  sim=0.763 anchors=4 (1867 1884 22 july)  p_xh: xh=0.79 en=0.38  modern: blind  oov=0.30

- xh: lomteto we Pasi basuke bawu nyatele nge nyawo becimile nganamabom. Ngenyaniso makube kuko i-ehe ngasele kwalento. Nasi eso saziso: — Isaziso se 4, 1884.] I Ofisi yo Mpatiswa.Bantsundu, E Kapa, 1 July, 1884. Ukutolikwa kwo Mteto we Pasi. Nkosi, — Njengokuba kubonakala nje ukuba amagwibi Omteto wa 22 womnyaka we 1867 akahlale egciniwe ezingqondweni okanye akaziwa kakuhle, ndiyalezwa ukuba mandalate
- en: “ something rotten in the state of Denmark ” : — Circular No. 4, 1884.] Office of the Secretary for Native Affairs, Cape Town, 1st July, 1884. INTERPRETATION OF THE PASS LAW. SIR, — As it appears that the provisions of Act No. 22 of 1867 are not always borne in mind or clearly understood, I am directed to point

### 1893-02-02_p003_xh_010 / 1893-02-02_p003_en_027  sim=0.734 anchors=5 (15 19 august cala xalanga)  p_xh: xh=0.90 en=0.45  modern: blind  oov=0.26

- xh: kwe Taitile kwabantsundu e Cala. Kananjalo ndikutumela nencwadi ka Mr. Figilan awayekubhalele yona yomhla we 15 August, eyona ncwadi yake (original) ibhekiswe kwisebe lobu Rulumeni elipete lomcimbi. Ndinembeko Nkosi, yokuba sicaka sako, R. ROSE-INNES, Umpatiswa Ongapantsi Wabantsundu. Kwi Sebe Lemihlaba, Imingxuma Nokulima, 19 November, 1892. Ukumiwa kwemihlatyana enikelwe Abantsundu bo Mntan' Omh
- en: Xalanga on the 19th ultimo, on the subject of granting land at Cala to Natives. I also enclose a copy of Mr. Figlan’s letter to you of the 15th August, the original having been referred officially to the Department dealing with the matter. I have the honour to be, Sir, Your obedient servant, Department of Lands and Mines and Agriculture,

### 1885-03-02_p003_xh_010 / 1885-03-02_p002_en_014  sim=0.733 anchors=2 (gladstone soudan)  p_xh: xh=0.55 en=0.35  modern: blind  oov=0.39

- xh: Zakuba zibuye zaya zikula ngokukula inkatazo Emantla e-Jeputa wabuya watunyelwa e-Soudan, — kanti ngeli ixa uyela umya-kanye ukuba angaze abuye abuye. Inxenye seyiyawaqala igxeka u-Mr. GLADSTONE imgxeka ngento angenanto nayo, intoke leyo ekupela engasuka ifanekiswe nalamkwa wasebubedenini wokuba kubeko bani unukwayo xa ngaba kuko umntu ofileyo, U-Rulumente wa Pesheya uyenzile eyake imfanelo ngokum
- en: own broth. Troubles having been brewing in Upper Egypt he was again sent to the Soudan — this time not to return again. Some are already blaming Mr. GLADSTONE | for this untoward event, but this can only be compared to the barbarous practice of the raw natives, who must have someone smelt out for the death of a person.

### 1886-08-25_p002_xh_029 / 1886-08-25_p003_en_009  sim=0.728 anchors=3 (760 herschel news)  p_xh: xh=0.61 en=0.24  modern: blind  oov=0.40

- xh: buvoti. Umteto umhlope ngalento abangawaziyo bangabuza nakulipina igqweta elipakati kwabo. Lenteto siyenza ngenxa yoku sikufunde epepeni lase Alvani i Border News eliti “siva ukuba amagama abantu abayi 760 abulewe eluhlwini lwaba voti kwisiqingata sase Gqili (Herschel).” Akuko ntandabuzo lawo magama anga 760 ngawabantu abamnyama. Siyakolwa ukuba lawo magama abuleweyo asaxonyiwe pambi kwe ofisi zem
- en: to know it may consult any attorney on the spot. We are led into these remarks by what we read in one of the Aliwal North papers (the Borden- News) which “understands that no less than 760 names have been struck off the list of voters in the Herschel Reserve.” There cannot be the slightest doubt but that the 760

