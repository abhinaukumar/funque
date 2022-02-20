dataset_name = 'MCL_V'
yuv_fmt = 'yuv420p'
width = 1920
height = 1080
quality_width = 1920
quality_height = 1080

path_to_dataset = '[path to dataset]'
ref_dir = path_to_dataset + '/reference_sequences/yuv'
dis_dir = path_to_dataset + '/distorted_sequences/yuv'

ref_videos = [{'content_id': 0,
               'content_name': 'BQTerrace',
               'path': ref_dir + '/BQTerrace_30fps.yuv'},
              {'content_id': 1,
               'content_name': 'BigBuckBunny',
               'path': ref_dir + '/BigBuckBunny_25fps.yuv'},
              {'content_id': 2,
               'content_name': 'BirdsInCage',
               'path': ref_dir + '/BirdsInCage_30fps.yuv'},
              {'content_id': 3,
               'content_name': 'CrowdRun',
               'path': ref_dir + '/CrowdRun_25fps.yuv'},
              {'content_id': 4,
               'content_name': 'DanceKiss',
               'path': ref_dir + '/DanceKiss_25fps.yuv'},
              {'content_id': 5,
               'content_name': 'ElFuente1',
               'path': ref_dir + '/ElFuente1_30fps.yuv'},
              {'content_id': 6,
               'content_name': 'ElFuente2',
               'path': ref_dir + '/ElFuente2_30fps.yuv'},
              {'content_id': 7,
               'content_name': 'FoxBird',
               'path': ref_dir + '/FoxBird_25fps.yuv'},
              {'content_id': 8,
               'content_name': 'Kimono1',
               'path': ref_dir + '/Kimono1_24fps.yuv'},
              {'content_id': 9,
               'content_name': 'OldTownCross',
               'path': ref_dir + '/OldTownCross_25fps.yuv'},
              {'content_id': 10,
               'content_name': 'Seeking',
               'path': ref_dir + '/Seeking_25fps.yuv'},
              {'content_id': 11,
               'content_name': 'Tennis',
               'path': ref_dir + '/Tennis_24fps.yuv'}]

dis_videos = [{'asset_id': 0,
               'content_id': 1,
               'dmos': 66.72413793103448,
               'path': dis_dir + '/BigBuckBunny_25fps_H264_4.yuv'},
              {'asset_id': 1,
               'content_id': 1,
               'dmos': 58.9655172413793,
               'path': dis_dir + '/BigBuckBunny_25fps_SDH264HD_4.yuv'},
              {'asset_id': 2,
               'content_id': 1,
               'dmos': 49.310344827586206,
               'path': dis_dir + '/BigBuckBunny_25fps_H264_3.yuv'},
              {'asset_id': 3,
               'content_id': 1,
               'dmos': 44.13793103448276,
               'path': dis_dir + '/BigBuckBunny_25fps_SDH264HD_3.yuv'},
              {'asset_id': 4,
               'content_id': 1,
               'dmos': 26.03448275862069,
               'path': dis_dir + '/BigBuckBunny_25fps_H264_2.yuv'},
              {'asset_id': 5,
               'content_id': 1,
               'dmos': 22.93103448275862,
               'path': dis_dir + '/BigBuckBunny_25fps_SDH264HD_2.yuv'},
              {'asset_id': 6,
               'content_id': 1,
               'dmos': 7.586206896551724,
               'path': dis_dir + '/BigBuckBunny_25fps_H264_1.yuv'},
              {'asset_id': 7,
               'content_id': 1,
               'dmos': 3.4482758620689657,
               'path': dis_dir + '/BigBuckBunny_25fps_SDH264HD_1.yuv'},
              {'asset_id': 8,
               'content_id': 2,
               'dmos': 70.6896551724138,
               'path': dis_dir + '/BirdsInCage_30fps_H264_4.yuv'},
              {'asset_id': 9,
               'content_id': 2,
               'dmos': 57.06896551724138,
               'path': dis_dir + '/BirdsInCage_30fps_SDH264HD_4.yuv'},
              {'asset_id': 10,
               'content_id': 2,
               'dmos': 53.44827586206897,
               'path': dis_dir + '/BirdsInCage_30fps_H264_3.yuv'},
              {'asset_id': 11,
               'content_id': 2,
               'dmos': 36.724137931034484,
               'path': dis_dir + '/BirdsInCage_30fps_SDH264HD_3.yuv'},
              {'asset_id': 12,
               'content_id': 2,
               'dmos': 34.310344827586206,
               'path': dis_dir + '/BirdsInCage_30fps_H264_2.yuv'},
              {'asset_id': 13,
               'content_id': 2,
               'dmos': 12.931034482758621,
               'path': dis_dir + '/BirdsInCage_30fps_SDH264HD_2.yuv'},
              {'asset_id': 14,
               'content_id': 2,
               'dmos': 11.03448275862069,
               'path': dis_dir + '/BirdsInCage_30fps_H264_1.yuv'},
              {'asset_id': 15,
               'content_id': 2,
               'dmos': 6.379310344827587,
               'path': dis_dir + '/BirdsInCage_30fps_SDH264HD_1.yuv'},
              {'asset_id': 16,
               'content_id': 0,
               'dmos': 69.65517241379311,
               'path': dis_dir + '/BQTerrace_30fps_H264_4.yuv'},
              {'asset_id': 17,
               'content_id': 0,
               'dmos': 56.55172413793103,
               'path': dis_dir + '/BQTerrace_30fps_SDH264HD_4.yuv'},
              {'asset_id': 18,
               'content_id': 0,
               'dmos': 53.62068965517241,
               'path': dis_dir + '/BQTerrace_30fps_H264_3.yuv'},
              {'asset_id': 19,
               'content_id': 0,
               'dmos': 38.793103448275865,
               'path': dis_dir + '/BQTerrace_30fps_SDH264HD_3.yuv'},
              {'asset_id': 20,
               'content_id': 0,
               'dmos': 31.551724137931036,
               'path': dis_dir + '/BQTerrace_30fps_H264_2.yuv'},
              {'asset_id': 21,
               'content_id': 0,
               'dmos': 18.793103448275865,
               'path': dis_dir + '/BQTerrace_30fps_SDH264HD_2.yuv'},
              {'asset_id': 22,
               'content_id': 0,
               'dmos': 8.96551724137931,
               'path': dis_dir + '/BQTerrace_30fps_H264_1.yuv'},
              {'asset_id': 23,
               'content_id': 0,
               'dmos': 3.4482758620689657,
               'path': dis_dir + '/BQTerrace_30fps_SDH264HD_1.yuv'},
              {'asset_id': 24,
               'content_id': 3,
               'dmos': 69.13793103448276,
               'path': dis_dir + '/CrowdRun_25fps_H264_4.yuv'},
              {'asset_id': 25,
               'content_id': 3,
               'dmos': 56.37931034482759,
               'path': dis_dir + '/CrowdRun_25fps_SDH264HD_4.yuv'},
              {'asset_id': 26,
               'content_id': 3,
               'dmos': 53.793103448275865,
               'path': dis_dir + '/CrowdRun_25fps_H264_3.yuv'},
              {'asset_id': 27,
               'content_id': 3,
               'dmos': 38.62068965517241,
               'path': dis_dir + '/CrowdRun_25fps_SDH264HD_3.yuv'},
              {'asset_id': 28,
               'content_id': 3,
               'dmos': 31.551724137931036,
               'path': dis_dir + '/CrowdRun_25fps_H264_2.yuv'},
              {'asset_id': 29,
               'content_id': 3,
               'dmos': 19.82758620689655,
               'path': dis_dir + '/CrowdRun_25fps_SDH264HD_2.yuv'},
              {'asset_id': 30,
               'content_id': 3,
               'dmos': 10.517241379310345,
               'path': dis_dir + '/CrowdRun_25fps_H264_1.yuv'},
              {'asset_id': 31,
               'content_id': 3,
               'dmos': 2.9310344827586206,
               'path': dis_dir + '/CrowdRun_25fps_SDH264HD_1.yuv'},
              {'asset_id': 32,
               'content_id': 4,
               'dmos': 66.2962962962963,
               'path': dis_dir + '/DanceKiss_25fps_H264_4.yuv'},
              {'asset_id': 33,
               'content_id': 4,
               'dmos': 62.03703703703703,
               'path': dis_dir + '/DanceKiss_25fps_SDH264HD_4.yuv'},
              {'asset_id': 34,
               'content_id': 4,
               'dmos': 46.11111111111111,
               'path': dis_dir + '/DanceKiss_25fps_H264_3.yuv'},
              {'asset_id': 35,
               'content_id': 4,
               'dmos': 46.2962962962963,
               'path': dis_dir + '/DanceKiss_25fps_SDH264HD_3.yuv'},
              {'asset_id': 36,
               'content_id': 4,
               'dmos': 25.925925925925924,
               'path': dis_dir + '/DanceKiss_25fps_H264_2.yuv'},
              {'asset_id': 37,
               'content_id': 4,
               'dmos': 23.518518518518515,
               'path': dis_dir + '/DanceKiss_25fps_SDH264HD_2.yuv'},
              {'asset_id': 38,
               'content_id': 4,
               'dmos': 3.5185185185185186,
               'path': dis_dir + '/DanceKiss_25fps_H264_1.yuv'},
              {'asset_id': 39,
               'content_id': 4,
               'dmos': 7.962962962962963,
               'path': dis_dir + '/DanceKiss_25fps_SDH264HD_1.yuv'},
              {'asset_id': 40,
               'content_id': 5,
               'dmos': 68.14814814814815,
               'path': dis_dir + '/ElFuente1_30fps_H264_4.yuv'},
              {'asset_id': 41,
               'content_id': 5,
               'dmos': 60.370370370370374,
               'path': dis_dir + '/ElFuente1_30fps_SDH264HD_4.yuv'},
              {'asset_id': 42,
               'content_id': 5,
               'dmos': 46.66666666666667,
               'path': dis_dir + '/ElFuente1_30fps_H264_3.yuv'},
              {'asset_id': 43,
               'content_id': 5,
               'dmos': 44.25925925925925,
               'path': dis_dir + '/ElFuente1_30fps_SDH264HD_3.yuv'},
              {'asset_id': 44,
               'content_id': 5,
               'dmos': 28.703703703703702,
               'path': dis_dir + '/ElFuente1_30fps_H264_2.yuv'},
              {'asset_id': 45,
               'content_id': 5,
               'dmos': 22.40740740740741,
               'path': dis_dir + '/ElFuente1_30fps_SDH264HD_2.yuv'},
              {'asset_id': 46,
               'content_id': 5,
               'dmos': 2.962962962962963,
               'path': dis_dir + '/ElFuente1_30fps_H264_1.yuv'},
              {'asset_id': 47,
               'content_id': 5,
               'dmos': 7.4074074074074066,
               'path': dis_dir + '/ElFuente1_30fps_SDH264HD_1.yuv'},
              {'asset_id': 48,
               'content_id': 6,
               'dmos': 64.25925925925925,
               'path': dis_dir + '/ElFuente2_30fps_H264_4.yuv'},
              {'asset_id': 49,
               'content_id': 6,
               'dmos': 61.11111111111111,
               'path': dis_dir + '/ElFuente2_30fps_SDH264HD_4.yuv'},
              {'asset_id': 50,
               'content_id': 6,
               'dmos': 45.0,
               'path': dis_dir + '/ElFuente2_30fps_H264_3.yuv'},
              {'asset_id': 51,
               'content_id': 6,
               'dmos': 44.25925925925925,
               'path': dis_dir + '/ElFuente2_30fps_SDH264HD_3.yuv'},
              {'asset_id': 52,
               'content_id': 6,
               'dmos': 28.518518518518515,
               'path': dis_dir + '/ElFuente2_30fps_H264_2.yuv'},
              {'asset_id': 53,
               'content_id': 6,
               'dmos': 24.629629629629626,
               'path': dis_dir + '/ElFuente2_30fps_SDH264HD_2.yuv'},
              {'asset_id': 54,
               'content_id': 6,
               'dmos': 7.037037037037037,
               'path': dis_dir + '/ElFuente2_30fps_H264_1.yuv'},
              {'asset_id': 55,
               'content_id': 6,
               'dmos': 4.444444444444445,
               'path': dis_dir + '/ElFuente2_30fps_SDH264HD_1.yuv'},
              {'asset_id': 56,
               'content_id': 8,
               'dmos': 69.25925925925925,
               'path': dis_dir + '/Kimono1_24fps_H264_4.yuv'},
              {'asset_id': 57,
               'content_id': 8,
               'dmos': 61.48148148148148,
               'path': dis_dir + '/Kimono1_24fps_SDH264HD_4.yuv'},
              {'asset_id': 58,
               'content_id': 8,
               'dmos': 48.51851851851852,
               'path': dis_dir + '/Kimono1_24fps_H264_3.yuv'},
              {'asset_id': 59,
               'content_id': 8,
               'dmos': 42.96296296296297,
               'path': dis_dir + '/Kimono1_24fps_SDH264HD_3.yuv'},
              {'asset_id': 60,
               'content_id': 8,
               'dmos': 26.11111111111111,
               'path': dis_dir + '/Kimono1_24fps_H264_2.yuv'},
              {'asset_id': 61,
               'content_id': 8,
               'dmos': 23.14814814814815,
               'path': dis_dir + '/Kimono1_24fps_SDH264HD_2.yuv'},
              {'asset_id': 62,
               'content_id': 8,
               'dmos': 7.962962962962963,
               'path': dis_dir + '/Kimono1_24fps_H264_1.yuv'},
              {'asset_id': 63,
               'content_id': 8,
               'dmos': 3.333333333333333,
               'path': dis_dir + '/Kimono1_24fps_SDH264HD_1.yuv'},
              {'asset_id': 64,
               'content_id': 7,
               'dmos': 65.76923076923077,
               'path': dis_dir + '/FoxBird_25fps_H264_4.yuv'},
              {'asset_id': 65,
               'content_id': 7,
               'dmos': 63.846153846153854,
               'path': dis_dir + '/FoxBird_25fps_SDH264HD_4.yuv'},
              {'asset_id': 66,
               'content_id': 7,
               'dmos': 43.46153846153846,
               'path': dis_dir + '/FoxBird_25fps_H264_3.yuv'},
              {'asset_id': 67,
               'content_id': 7,
               'dmos': 45.96153846153846,
               'path': dis_dir + '/FoxBird_25fps_SDH264HD_3.yuv'},
              {'asset_id': 68,
               'content_id': 7,
               'dmos': 28.846153846153847,
               'path': dis_dir + '/FoxBird_25fps_H264_2.yuv'},
              {'asset_id': 69,
               'content_id': 7,
               'dmos': 15.0,
               'path': dis_dir + '/FoxBird_25fps_SDH264HD_2.yuv'},
              {'asset_id': 70,
               'content_id': 7,
               'dmos': 16.153846153846153,
               'path': dis_dir + '/FoxBird_25fps_H264_1.yuv'},
              {'asset_id': 71,
               'content_id': 7,
               'dmos': 0.7692307692307693,
               'path': dis_dir + '/FoxBird_25fps_SDH264HD_1.yuv'},
              {'asset_id': 72,
               'content_id': 9,
               'dmos': 70.76923076923077,
               'path': dis_dir + '/OldTownCross_25fps_H264_4.yuv'},
              {'asset_id': 73,
               'content_id': 9,
               'dmos': 57.11538461538461,
               'path': dis_dir + '/OldTownCross_25fps_SDH264HD_4.yuv'},
              {'asset_id': 74,
               'content_id': 9,
               'dmos': 55.76923076923077,
               'path': dis_dir + '/OldTownCross_25fps_H264_3.yuv'},
              {'asset_id': 75,
               'content_id': 9,
               'dmos': 40,
               'path': dis_dir + '/OldTownCross_25fps_SDH264HD_3.yuv'},
              {'asset_id': 76,
               'content_id': 9,
               'dmos': 32.30769230769231,
               'path': dis_dir + '/OldTownCross_25fps_H264_2.yuv'},
              {'asset_id': 77,
               'content_id': 9,
               'dmos': 19.038461538461537,
               'path': dis_dir + '/OldTownCross_25fps_SDH264HD_2.yuv'},
              {'asset_id': 78,
               'content_id': 9,
               'dmos': 9.230769230769232,
               'path': dis_dir + '/OldTownCross_25fps_H264_1.yuv'},
              {'asset_id': 79,
               'content_id': 9,
               'dmos': 2.5,
               'path': dis_dir + '/OldTownCross_25fps_SDH264HD_1.yuv'},
              {'asset_id': 80,
               'content_id': 10,
               'dmos': 68.07692307692308,
               'path': dis_dir + '/Seeking_25fps_H264_4.yuv'},
              {'asset_id': 81,
               'content_id': 10,
               'dmos': 61.73076923076923,
               'path': dis_dir + '/Seeking_25fps_SDH264HD_4.yuv'},
              {'asset_id': 82,
               'content_id': 10,
               'dmos': 49.03846153846154,
               'path': dis_dir + '/Seeking_25fps_H264_3.yuv'},
              {'asset_id': 83,
               'content_id': 10,
               'dmos': 40.38461538461539,
               'path': dis_dir + '/Seeking_25fps_SDH264HD_3.yuv'},
              {'asset_id': 84,
               'content_id': 10,
               'dmos': 30,
               'path': dis_dir + '/Seeking_25fps_H264_2.yuv'},
              {'asset_id': 85,
               'content_id': 10,
               'dmos': 21.346153846153847,
               'path': dis_dir + '/Seeking_25fps_SDH264HD_2.yuv'},
              {'asset_id': 86,
               'content_id': 10,
               'dmos': 8.653846153846153,
               'path': dis_dir + '/Seeking_25fps_H264_1.yuv'},
              {'asset_id': 87,
               'content_id': 10,
               'dmos': 3.076923076923077,
               'path': dis_dir + '/Seeking_25fps_SDH264HD_1.yuv'},
              {'asset_id': 88,
               'content_id': 11,
               'dmos': 69.23076923076923,
               'path': dis_dir + '/Tennis_24fps_H264_4.yuv'},
              {'asset_id': 89,
               'content_id': 11,
               'dmos': 61.92307692307693,
               'path': dis_dir + '/Tennis_24fps_SDH264HD_4.yuv'},
              {'asset_id': 90,
               'content_id': 11,
               'dmos': 45.19230769230769,
               'path': dis_dir + '/Tennis_24fps_H264_3.yuv'},
              {'asset_id': 91,
               'content_id': 11,
               'dmos': 43.26923076923077,
               'path': dis_dir + '/Tennis_24fps_SDH264HD_3.yuv'},
              {'asset_id': 92,
               'content_id': 11,
               'dmos': 26.538461538461537,
               'path': dis_dir + '/Tennis_24fps_H264_2.yuv'},
              {'asset_id': 93,
               'content_id': 11,
               'dmos': 25.76923076923077,
               'path': dis_dir + '/Tennis_24fps_SDH264HD_2.yuv'},
              {'asset_id': 94,
               'content_id': 11,
               'dmos': 3.4615384615384617,
               'path': dis_dir + '/Tennis_24fps_H264_1.yuv'},
              {'asset_id': 95,
               'content_id': 11,
               'dmos': 6.923076923076923,
               'path': dis_dir + '/Tennis_24fps_SDH264HD_1.yuv'}]