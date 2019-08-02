# apt install openalpr
# apt install python3-openalpr

from openalpr import Alpr
import cv2
import sys
import json

if len(sys.argv)!=2:
    print('Usage:',sys.argv[0],'<car image>')
    sys.exit(2)

alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

alpr.set_top_n(10)
alpr.set_default_region("md")

results = alpr.recognize_file(sys.argv[1])

if results is not None:
    print('Found plate',len(results['results']))

    # load iamge
    img = cv2.imread(sys.argv[1])

    for plate in results['results']:
        print('Plate match coordinates:')
        print(plate['coordinates'][0]['x'],plate['coordinates'][0]['y'])
        print(plate['coordinates'][1]['x'],plate['coordinates'][1]['y'])
        print(plate['coordinates'][2]['x'],plate['coordinates'][2]['y'])
        print(plate['coordinates'][3]['x'],plate['coordinates'][3]['y'])
        print()

        # draw box around plate
        x1,y1=plate['coordinates'][0]['x'],plate['coordinates'][0]['y']
        x2,y2=plate['coordinates'][2]['x'],plate['coordinates'][2]['y']
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        print('   {:12s} {:12s}'.format('Plate', 'Confidence'))
        for candidate in plate['candidates']:
            prefix = "-"
            if candidate['matches_template']:
                prefix = "*"

            print('  {} {:12s}{:12f}'.format(prefix, candidate['plate'], candidate['confidence']))

cv2.imshow('Car Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Call when completely done to release memory
#alpr.unload()
