import sys,os
import argparse
import requests
import random
import pandas as pd


allowed_classes = ['motorway','trunk','primary','secondary','tertiary','street','pedestrian','track','service','path']
weights = [1,1,1,1,10,100,1000,1000,1000,1000]

def request_tile(url):
    #print(f'Getting tile {url}')
    r = requests.get(url)
    r.raise_for_status()
    return r

def collapseAndSampleSchools(schoolsDict,sample):

    d = {}
    d2 = {}
    used = []
    num_keys = 0
    for key1 in schoolsDict:
        if key1 not in used:
            (lat1, lon1) = schoolsDict[key1]
            collapse = True
            for key2 in schoolsDict:
                if key1!=key2:
                    (lat2, lon2) = schoolsDict[key2]
                    if lat1==lat2 and lon1==lon2:
                        used.append(key2)
                        collapse = False
                        break
            if collapse:
                d[key1] = (lat1,lon1)

                if num_keys<sample:
                    rd = random.random()
                    if rd < 0.2:
                        d2[key1] = (lat1,lon1)
                        num_keys += 1

    if sample==0:
        return d
    return d2

def getFeatures(school,radius,limit,token):
    lat,lon = school
    url = f'https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/tilequery/{lon},{lat}.json?radius={radius}&limit={limit}&dedupe&geometry=linestring&access_token={token}'

    response = request_tile(url).json()
    if 'features' in response:
        roads = response['features']
    else:
        print("Issue creating isochrone, skipping", response)
        exit()

    typeToDist = {}

    for i in range(len(roads)):
        properties = roads[i]['properties']
        if properties['tilequery']['layer']=='road':
            clas = properties['class']
            if clas in allowed_classes:
                dist = int(properties['tilequery']['distance'])
                if clas not in typeToDist:
                    typeToDist[clas] = dist
                elif dist < typeToDist[clas]:
                    typeToDist[clas] = dist

    return(typeToDist)

def main():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--school-data', '-sd',
                          help='csv file containing school data', required=True)
    required.add_argument('--token', '-t', 
                          help='mapbox token', required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--radius", '-r', type=int, default=10000,
                          help='distance in meters to search for roads around school')
    optional.add_argument("--limit", '-l', type=int, default=50,
                          help='limit of roads to find')
    optional.add_argument("--sample", '-s', type=int, default=100,
                          help='sample of shcools, for testing')
    optional.add_argument("--schools-id", '-si', default='giga_school_id', help='school id field name')
    optional.add_argument("--schools-lat", '-sla', default='latitude', help='school latitude field name')
    optional.add_argument("--schools-lon", '-slo', default='longitude', help='school longitude field name')

    args = parser.parse_args()
    fn = args.school_data
    radius = args.radius
    limit = args.limit
    token = args.token
    sample = args.sample
    id = args.schools_id
    slat = args.schools_lat
    slon = args.schools_lon

    schools = pd.read_csv(fn)

    schoolsDict = {id: (lat, lon) for id, lat, lon in zip(schools[id], schools[slat], schools[slon])}

    schoolsDictSampled = collapseAndSampleSchools(schoolsDict,sample)

    schoolToFeatures = {}

    for key in schoolsDictSampled:
        features = getFeatures(schoolsDictSampled[key],radius,limit,token)
        schoolToFeatures[key] = features

    title = 'School id,lat,lon,'
    for i in range(len(allowed_classes)):
        title+=allowed_classes[i]+','
    title += 'min_score'
    print(title)

    for key in schoolToFeatures:
        lat,lon = schoolsDictSampled[key]
        d = schoolToFeatures[key]
        pline = key + ',' + str(lat) + ',' + str(lon) + ','
        min_score = 1000000
        for i in range(len(allowed_classes)):
            if allowed_classes[i] in d:
                dist = d[allowed_classes[i]]
                score = dist*weights[i]
                if score<min_score:
                    min_score = score
            else:
                dist = -1
            pline += str(dist)+','
        pline += str(min_score)
        print(pline)






if __name__ == "__main__":
    main()
