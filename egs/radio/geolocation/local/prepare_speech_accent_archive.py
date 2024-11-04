import argparse
from geopy import geocoders
from geopy.geocoders import Nominatim
from lhotse import (
    CutSet, SupervisionSet, RecordingSet, SupervisionSegment, fix_manifests
)
import csv
from tqdm import tqdm
import json


def get_geolocations(args):
    geolocator = Nominatim(user_agent="myapplication")
    with open(args.metadata) as f:
        csv_reader = csv.reader(f)
        data = [l for l in csv_reader]
    
    headers = data[0]
    data_dict = {}
    for l in tqdm(data[1:]):
        if "synthesized" in l[3]:
            continue
        if l[3] not in data_dict:
            data_dict[l[3]] = {}
        for i in range(len(headers)):
            if headers[i] == "birthplace":
                try:
                    coords = geolocator.geocode(l[i], timeout=None)[1]
                except TypeError:
                    if l[i] == "ramun, israel (occupied territory)":
                        coords = geolocator.geocode("rammun", timeout=None)[1]
                    else:
                        try:
                            coords = geolocator.geocode(l[7], timeout=None)[1]
                        except:
                            print(l[i])
                            import pdb; pdb.set_trace()

                data_dict[l[3]][headers[i]] = coords
            else:
                data_dict[l[3]][headers[i]] = l[i]
        
    file2location = {}
    for k in data_dict:
        file2location[k] = data_dict[k]['birthplace']

    with open('speech_accent_archive_locations.json', 'w') as f:
        json.dump(file2location, f, indent=4)
    
    return file2location, data_dict


def main(args):
    if args.geolocations is None:
        file2location, data_dict = get_geolocations(args)
    else:
        file2location = json.load(open(args.geolocations))
   
    recos = RecordingSet.from_dir(args.recos, "*.mp3") 
    segs = []
    for r in recos:
        if r.id not in file2location:
            continue
        lat, lon = file2location[r.id]
        segs.append(
            SupervisionSegment(
                start=0,
                duration=r.duration,
                recording_id=r.id,
                id=r.id,
                custom={'lat': lat, 'lon': lon}
            )
        )
    sups = SupervisionSet.from_segments(segs)
    recos, segs = fix_manifests(recos, segs)
    cuts = CutSet.from_manifests(recordings=recos, supervisions=sups) 
    cuts = cuts.resample(16000)
    with CutSet.open_writer(args.out) as cut_writer:
        for c in tqdm(cuts, "Writing cuts..."):
            cut_writer.write(c)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recos")
    parser.add_argument("metadata")
    parser.add_argument("out")
    parser.add_argument("--geolocations")
    args = parser.parse_args()
    main(args)
