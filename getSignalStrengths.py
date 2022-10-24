import sys,os
import math
import pandas as pd
import xlsxwriter
import argparse
import geopandas as gpd

def readSchools(fn):
    if not (os.path.isfile(fn) and os.access(fn, os.R_OK)):
        return True
    df = pd.read_excel(fn, engine='openpyxl')
    rows, cols = df.shape

    ids = list(map(str, list(df['idemis_code'])))
    lats = list(map(float, list(df['Lat'])))
    lons = list(map(float, list(df['Lon'])))

    schools = {}
    for i in range(len(ids)):
        schools[ids[i]] = (lons[i],lats[i])

    return schools

def Haversine(coord1, coord2):
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
         math.cos(phi_1) * math.cos(phi_2) * \
         math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers
    miles = meters * 0.000621371  # output distance in miles
    feet = miles * 5280  # output distance in feet

    return meters

def f(coord1,coord2,range,radio,k,l,j,coverage):
    d = Haversine(coord1, coord2)
    if d<=range+coverage:
        if radio=='GSM':
            return l*(20*math.log10(d)+11)
        if radio=='UMTS':
            return k*(20 * math.log10(d) + 11)
        if radio=='LTE':
            return j*(20 * math.log10(d) + 11)
        return 0 #is there another radio type?
    return 0

def g(coord1,coord2,radio):
    if radio=='UMTS':
        return Haversine(coord1, coord2)

    return 1000000

def h(coord1,coord2,radio):
    if radio=='GSM':
        return Haversine(coord1, coord2)

    return 1000000

def i(coord1,coord2,radio):
    if radio=='LTE':
        return Haversine(coord1, coord2)

    return 1000000


def getStrengthInLocation(coordSchool,cellTowers,k,l,j,coverage):
    result = [f((x, y), coordSchool, z, v, k, l, j, coverage) for x, y, z, v in zip(cellTowers['lon'], cellTowers['lat'], cellTowers['range'], cellTowers['radio'])]
    return sum(result)

def getDist2umtsInLocation(coordSchool,cellTowers):
    result = [g((x, y), coordSchool, z) for x, y, z in zip(cellTowers['lon'], cellTowers['lat'], cellTowers['radio'])]
    return min(result)

def getDist2gsmInLocation(coordSchool,cellTowers):
    result = [h((x, y), coordSchool, z) for x, y, z in zip(cellTowers['lon'], cellTowers['lat'], cellTowers['radio'])]
    return min(result)

def getDist2lteInLocation(coordSchool,cellTowers):
    result = [i((x, y), coordSchool, z) for x, y, z in zip(cellTowers['lon'], cellTowers['lat'], cellTowers['radio'])]
    return min(result)

def electricity(r,w):
    if r=='No':
        return 0
    return w

def getScores(cellTowers,schools,k,l,j,m,coverage):
    scores = [electricity(z,m) + getStrengthInLocation((x,y),cellTowers,k,l,j,coverage) for x,y,z in zip(schools['longitude'],schools['latitude'],schools['electricity_availability'])]
    dist2umts = [getDist2umtsInLocation((x, y), cellTowers) for x, y in
              zip(schools['longitude'], schools['latitude'])]
    dist2gsm = [getDist2gsmInLocation((x, y), cellTowers) for x, y in
              zip(schools['longitude'], schools['latitude'])]
    dist2lte = [getDist2lteInLocation((x, y), cellTowers) for x, y in
                zip(schools['longitude'], schools['latitude'])]

    return scores, dist2umts, dist2gsm,dist2lte

def getScoresOld2(cellTowers,schools,k,l,coverage):
    scores = [getStrengthInLocation((x,y),cellTowers,k,l,coverage) for x,y in zip(schools['longitude'],schools['latitude'])]
    dist2umts = [getDist2umtsInLocation((x, y), cellTowers) for x, y in
              zip(schools['longitude'], schools['latitude'])]
    dist2gsm = [getDist2gsmInLocation((x, y), cellTowers) for x, y in
              zip(schools['longitude'], schools['latitude'])]
    return scores,dist2umts,dist2gsm

def getScoresOld(cellTowers,schools,k,l):
    scores = {}
    for key in schools:
        coordSchool = schools[key]
        scores[key] = getStrengthInLocation(coordSchool,cellTowers,k,l)
    return scores

def main():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--school-data', '-sd', default='../../data/mapping/Botswana_school_processed.csv',help='csv file containing school data')#, required=True)
    required.add_argument('--celltower-data', '-ctd', default='../../data/opencellid/Botswana/btw_cell_towers.csv',
                          help='csv or shape file with cell tower data')  # ,required = True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--LTE-multiplier", '-j', type=int, default=100, help='multiplier constant for LTE')
    optional.add_argument("--UMTS-multiplier", '-k', type=int, default=10, help='multiplier constant for UMTS')
    optional.add_argument("--GSM-multiplier", '-l', type=int, default=1, help='multiplier constant for GSM')
    optional.add_argument("--Electricity-multiplier", '-m', type=int, default=1000, help='multiplier constant for GSM')
    optional.add_argument("--schools-id", '-si', default='giga_id_school', help='school id field name')
    optional.add_argument("--coverage-range","-cr",type=int, default=10000, help='coverage range in meters')
    optional.add_argument('--output', '-o', default='../../results/Botswana_school_infrastructure_score.csv',help='output file name')
    args = parser.parse_args()

    toKeep = ['electricity_availability', 'dist2umts', 'dist2gsm', 'dist2lte', 'infrastructureScore']
    toKeep.append(args.schools_id)

    schools = pd.read_csv(args.school_data)

    if args.celltower_data[-3:]=='shp':
        cellTowers = gpd.read_file(args.celltower_data)
    else:
        cellTowers = pd.read_csv(args.celltower_data)
    k = args.UMTS_multiplier
    l = args.GSM_multiplier
    j = args.LTE_multiplier
    m = args.Electricity_multiplier
    schoolScores, dist2umts, dist2gsm,dist2lte = getScores(cellTowers, schools, k, l, j, m, args.coverage_range)

    schools['dist2umts'] = dist2umts
    schools['dist2gsm'] = dist2gsm
    schools['dist2lte'] = dist2lte
    schools['infrastructureScore'] = schoolScores

    cols = list(schools)
    for col in cols:
        if col not in toKeep:
            schools.drop(col, inplace=True, axis=1)
    schools.to_csv(args.output)

if __name__ == "__main__":
    main()