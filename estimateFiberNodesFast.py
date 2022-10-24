import sys,os
import pandas as pd

import argparse

from decimal import Decimal
from math import cos, sin, sqrt
import math
import numpy as np
import random as rd

R = 6371000 #earth radius in meters

def toGecocentric(p1,R):
    x_p1 = R * Decimal(cos(math.radians(p1[1])) * cos(math.radians(p1[0])))  # x = cos(lon)*cos(lat)
    y_p1 = R * Decimal(sin(math.radians(p1[1])) * cos(math.radians(p1[0])))  # y = sin(lon)*cos(lat)
    z_p1 = R * Decimal(sin(math.radians(p1[0])))  # z = sin(lat)

    return((x_p1,y_p1,z_p1))

def toLatLon(p):
    x,y,z=p
    plat = math.degrees(math.asin(z))
    plon = math.degrees(math.atan2(y, x))

    return (plat,plon)

'''
FINDING THE INTERSECTION COORDINATES (LAT/LON) OF TWO CIRCLES (GIVEN THE COORDINATES OF THE CENTER AND THE RADII)

The code below is based on whuber's brilliant work here:
https://gis.stackexchange.com/questions/48937/calculating-intersection-of-two-circles 

The idea is that;
  1. The points in question are the mutual intersections of three spheres: a sphere centered beneath location x1 (on the 
  earth's surface) of a given radius, a sphere centered beneath location x2 (on the earth's surface) of a given radius, and
  the earth itself, which is a sphere centered at O = (0,0,0) of a given radius.
  2. The intersection of each of the first two spheres with the earth's surface is a circle, which defines two planes.
  The mutual intersections of all three spheres therefore lies on the intersection of those two planes: a line.
  Consequently, the problem is reduced to intersecting a line with a sphere.

Note that "Decimal" is used to have higher precision which is important if the distance between two points are a few
meters.
'''


def intersection(x1, r1, x2, r2):
    # p1 = Coordinates of Point 1: latitude, longitude. This serves as the center of circle 1. Ex: (36.110174,  -90.953524)
    # r1_meter = Radius of circle 1 in meters
    # p2 = Coordinates of Point 2: latitude, longitude. This serves as the center of circle 1. Ex: (36.110174,  -90.953524)
    # r2_meter = Radius of circle 2 in meters
    '''
    1. Convert (lat, lon) to (x,y,z) geocentric coordinates.
    As usual, because we may choose units of measurement in which the earth has a unit radius
    '''
    #x_p1 = Decimal(cos(math.radians(p1[1]))*cos(math.radians(p1[0])))  # x = cos(lon)*cos(lat)
    #y_p1 = Decimal(sin(math.radians(p1[1]))*cos(math.radians(p1[0])))  # y = sin(lon)*cos(lat)
    #z_p1 = Decimal(sin(math.radians(p1[0])))                           # z = sin(lat)
    x_p1,y_p1,z_p1 = x1

    #x_p2 = Decimal(cos(math.radians(p2[1]))*cos(math.radians(p2[0])))  # x = cos(lon)*cos(lat)
    #y_p2 = Decimal(sin(math.radians(p2[1]))*cos(math.radians(p2[0])))  # y = sin(lon)*cos(lat)
    #z_p2 = Decimal(sin(math.radians(p2[0])))                           # z = sin(lat)
    x_p2, y_p2, z_p2 = x2
    '''
    2. Convert the radii r1 and r2 (which are measured along the sphere) to angles along the sphere.
    By definition, one nautical mile (NM) is 1/60 degree of arc (which is pi/180 * 1/60 = 0.0002908888 radians).
    '''
    #r1 = Decimal(math.radians((r1_meter/1852) / 60)) # r1_meter/1852 converts meter to Nautical mile.
    #r2 = Decimal(math.radians((r2_meter/1852) / 60))
    '''
    3. The geodesic circle of radius r1 around x1 is the intersection of the earth's surface with an Euclidean sphere
    of radius sin(r1) centered at cos(r1)*x1.

    4. The plane determined by the intersection of the sphere of radius sin(r1) around cos(r1)*x1 and the earth's surface
    is perpendicular to x1 and passes through the point cos(r1)x1, whence its equation is x.x1 = cos(r1)
    (the "." represents the usual dot product); likewise for the other plane. There will be a unique point x0 on the
    intersection of those two planes that is a linear combination of x1 and x2. Writing x0 = ax1 + b*x2 the two planar
    equations are;
       cos(r1) = x.x1 = (a*x1 + b*x2).x1 = a + b*(x2.x1)
       cos(r2) = x.x2 = (a*x1 + b*x2).x2 = a*(x1.x2) + b
    Using the fact that x2.x1 = x1.x2, which I shall write as q, the solution (if it exists) is given by
       a = (cos(r1) - cos(r2)*q) / (1 - q^2),
       b = (cos(r2) - cos(r1)*q) / (1 - q^2).
    '''
    q = Decimal(np.dot(x1, x2))

    if q**2 != 1 :
        a = (Decimal(cos(r1)) - Decimal(cos(r2))*q) / (1 - q**2)
        b = (Decimal(cos(r2)) - Decimal(cos(r1))*q) / (1 - q**2)
        '''
        5. Now all other points on the line of intersection of the two planes differ from x0 by some multiple of a vector
        n which is mutually perpendicular to both planes. The cross product  n = x1~Cross~x2  does the job provided n is 
        nonzero: once again, this means that x1 and x2 are neither coincident nor diametrically opposite. (We need to 
        take care to compute the cross product with high precision, because it involves subtractions with a lot of
        cancellation when x1 and x2 are close to each other.)
        '''
        n = np.cross(x1, x2)
        '''
        6. Therefore, we seek up to two points of the form x0 + t*n which lie on the earth's surface: that is, their length
        equals 1. Equivalently, their squared length is 1:  
        1 = squared length = (x0 + t*n).(x0 + t*n) = x0.x0 + 2t*x0.n + t^2*n.n = x0.x0 + t^2*n.n
        '''
        x0_1 = [a*f for f in x1]
        x0_2 = [b*f for f in x2]
        x0 = [sum(f) for f in zip(x0_1, x0_2)]
        '''
          The term with x0.n disappears because x0 (being a linear combination of x1 and x2) is perpendicular to n.
          The two solutions easily are   t = sqrt((1 - x0.x0)/n.n)    and its negative. Once again high precision
          is called for, because when x1 and x2 are close, x0.x0 is very close to 1, leading to some loss of
          floating point precision.
        '''

        x0dx0 = np.dot(x0, x0)
        ndn = np.dot(n,n)
        if (x0dx0 <= 1) & (ndn != 0): # This is to secure that (1 - np.dot(x0, x0)) / np.dot(n,n) > 0
            t = Decimal(sqrt((1 - np.dot(x0, x0)) / np.dot(n,n)))
            t1 = t
            t2 = -t

            i1 = x0 + t1*n
            i2 = x0 + t2*n
            '''
            7. Finally, we may convert these solutions back to (lat, lon) by converting geocentric (x,y,z) to geographic
            coordinates. For the longitude, use the generalized arctangent returning values in the range -180 to 180
            degrees (in computing applications, this function takes both x and y as arguments rather than just the
            ratio y/x; it is sometimes called "ATan2").
            '''

            i1_lat = math.degrees( math.asin(i1[2]))
            i1_lon = math.degrees( math.atan2(i1[1], i1[0] ) )
            ip1 = (i1_lat, i1_lon)

            i2_lat = math.degrees( math.asin(i2[2]))
            i2_lon = math.degrees( math.atan2(i2[1], i2[0] ) )
            ip2 = (i2_lat, i2_lon)
            return [tuple(i1), tuple(i2)]
        elif (np.dot(n,n) == 0):
            return [] #return("The centers of the circles can be neither the same point nor antipodal points.")
        else:
            return []#return("The circles do not intersect")
    else:
        return [] #return("The centers of the circles can be neither the same point nor antipodal points.")



def Haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

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
    #km = meters / 1000.0  # output distance in kilometers
    #miles = meters * 0.000621371  # output distance in miles
    #feet = miles * 5280  # output distance in feet

    return meters


def Euclid(p1,p2,R):
    x_p1,y_p1,z_p1 = p1
    x_p2,y_p2,z_p2 = p2
    return math.sqrt((x_p1*R - x_p2*R) * (x_p1*R - x_p2*R) + (y_p1*R - y_p2*R) * (y_p1*R - y_p2*R) + (z_p1*R - z_p2*R) * (z_p1*R - z_p2*R))

def isSamePoint(p1,p2,R,offset):
    if p1==p2:
        return True
    d = Euclid(p1,p2,R)
    if d<=offset:
        return True
    return False

def centroid(p,ps):
    x,y,z = p
    xs = x
    ys = y
    zs = z
    n = 1.0
    for pi in ps:
        x,y,z = pi
        xs+=x
        ys+=y
        zs+=z
        n+=1.0

    n = Decimal(n)
    return (xs/n,ys/n,zs/n)

def centroidCoords(p,ps):
    lat,lon = p
    x = Decimal(cos(math.radians(lon))*cos(math.radians(lat)))  # x = cos(lon)*cos(lat)
    y = Decimal(sin(math.radians(lon))*cos(math.radians(lat)))  # y = sin(lon)*cos(lat)
    z = Decimal(sin(math.radians(lat)))                           # z = sin(lat)

    xs = x
    ys = y
    zs = z
    n = 1.0
    for pi in ps:
        lat, lon = p
        x = Decimal(cos(math.radians(lon)) * cos(math.radians(lat)))  # x = cos(lon)*cos(lat)
        y = Decimal(sin(math.radians(lon)) * cos(math.radians(lat)))  # y = sin(lon)*cos(lat)
        z = Decimal(sin(math.radians(lat)))  # z = sin(lat)
        xs+=x
        ys+=y
        zs+=z
        n+=1.0

    n = Decimal(n)
    return (xs/n,ys/n,zs/n)


def getFinalPoints(points,samePoint):
    pointsF = []
    for p in points:
        pc = centroid(p,samePoint[p])
        pointsF.append(toLatLon(pc))

    return pointsF

def validatePoints(schoolsInGeo,offset,R,vs):
    schoolsLeft = set()
    for s in schoolsInGeo:
        schoolsLeft.add(s)
    points = []
    used = []
    samePoint = {}

    for s1 in schoolsInGeo:
        if s1 in schoolsLeft:
            for s2 in schoolsInGeo:
                if s1!=s2 and s2 in schoolsLeft and (s1,s2) not in used:
                    removeG = False
                    p1,r1 = schoolsInGeo[s1]
                    p2,r2 = schoolsInGeo[s2]
                    candidates = intersection(p1,r1,p2,r2)
                    used.append((s1,s2))
                    used.append((s2,s1))

                    if len(candidates)>0:
                        for c in candidates:
                            samePoint[c] = []
                        for s3 in schoolsInGeo:
                            if s3!=s2 and s3!=s1 and s3 in schoolsLeft and (s1,s3) not in used and (s2,s3) not in used:
                                p3,r3 = schoolsInGeo[s3]
                                candidates2 = intersection(p1, r1, p3, r3)
                                used.append((s1, s2))
                                used.append((s2, s1))
                                if len(candidates2)==0:
                                    candidates2 = intersection(p2, r2, p3, r3)
                                if len(candidates2) > 0:
                                    remove = False
                                    for c1 in candidates:
                                        for c2 in candidates2:
                                            if isSamePoint(c1,c2,R,offset):
                                                samePoint[c1].append(c2)
                                                remove = True
                                                break
                                        if remove:
                                            break

                                    if remove:
                                        schoolsLeft.remove(s3)

                        for c in candidates:
                            if len(samePoint[c])>vs-3:
                                points.append(c)
                                removeG = True
                    if removeG:
                        schoolsLeft.remove(s1)
                        schoolsLeft.remove(s2)
                        break

    pointsF = getFinalPoints(points,samePoint)

    return pointsF,schoolsLeft



def collapseAndGeoSchools(schoolsDict):
    d = {}
    d2 = {}
    used = []
    for key1 in schoolsDict:
        if key1 not in used:
            (lat1, lon1, dist1) = schoolsDict[key1]
            collapse = True
            for key2 in schoolsDict:
                if key1!=key2:
                    (lat2, lon2, dist2) = schoolsDict[key2]
                    if lat1==lat2 and lon1==lon2:
                        used.append(key2)
                        collapse = False
                        break
            if collapse:
                d[key1] = (lat1,lon1,dist1)
                d2[key1] = (toGecocentric((lat1,lon1),1),Decimal(math.radians(((dist1*1000)/1852) / 60)))

    return d,d2

def doTestSet(schoolsDictCollapsed,schoolsInGeo):
    points = []
    pointsGeo = []
    ids = []

    i = 0
    for key in schoolsInGeo:
        if rd.random()>0.8:
            lat,lon,dist = schoolsDictCollapsed[key]
            p,dist2 = schoolsInGeo[key]
            inse = True
            for p2 in points:
                if Haversine((lat,lon),p2)<50000:
                    inse = False
            if inse:
                points.append((lat,lon))
                pointsGeo.append(p)
                ids.append(key)
                i += 1
                if i==3:
                    break



    works = False
    while not works:
        pointUsed = [[], [], []]
        d = {}
        d2 = {}
        works = True
        j = 0
        for key in schoolsInGeo:
            if key not in ids and rd.random()>0.8:
                dists = []
                lat,lon,di = schoolsDictCollapsed[key]
                p,di2 = schoolsInGeo[key]
                for i in range(len(points)):
                    dists.append(Haversine((lat,lon),points[i]))
                mindist = min(dists)
                argmindist = np.argmin(dists)
                pointUsed[argmindist].append(key)
                d[key] = (p,Decimal(math.radians(((mindist)/1852) / 60)))
                d2[key] = mindist
                j+=1
                if j==12:
                    break
        for p in pointUsed:
            if len(p)<3:
                works = False
    print(pointUsed)
    return d,d2,points,pointsGeo

def collapsePoints(candidatePoints,minimal_spacing):
    clusters = {}
    inCluster = {}
    for i in range(len(candidatePoints)):
        p1 = candidatePoints[i]
        if p1 not in inCluster:
            clusters[p1] = []
        for j in range(i+1,len(candidatePoints)):
            p2 = candidatePoints[j]
            if Haversine(p1,p2)<=minimal_spacing:
                if p1 in clusters:
                    clusters[p1].append(p2)
                    inCluster[p2] = p1
                elif p1 in inCluster:
                    clusters[inCluster[p1]].append(p2)
                    inCluster[p2] = inCluster[p1]
                else:
                    clusters[p1] = [p2]

    pointsF = []
    for p in clusters:
        pc = centroidCoords(p, clusters[p])
        pointsF.append(toLatLon(pc))

    return pointsF


def main():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--school-data', '-sd',help='csv file containing school data', required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--Offset", '-o', type=int, default=1000, help='offset distance in meters, error allowed in validating nodes')
    optional.add_argument("--minimal-spacing", '-ms', type=int, default=1000, help='minimal spacing between nodes in meters')
    optional.add_argument("--validating-schools", '-vs', type=int, default=3, help='minimum number of schools needed to validate node')
    optional.add_argument("--schools-id", '-si', default='giga_school_id', help='school id field name')
    optional.add_argument("--schools-lat", '-sla', default='latitude', help='school latitude field name')
    optional.add_argument("--schools-lon", '-slo', default='longitude', help='school longitude field name')
    optional.add_argument("--schools-fiber", '-sf', default='fiber_node_distance', help='school distance to fiber field name')
    args = parser.parse_args()

    fn = args.school_data
    offset = args.Offset
    minimal_spacing = args.minimal_spacing
    vs = args.validating_schools
    id = args.schools_id
    slat = args.schools_lat
    slon = args.schools_lon
    sfiber = args.schools_fiber
    schools = pd.read_csv(fn)


    schoolsDict = {id:(lat,lon,dist) for id,lat,lon,dist in zip(schools[id],schools[slat],schools[slon],schools[sfiber])}

    schoolsDictCollapsed, schoolsInGeo = collapseAndGeoSchools(schoolsDict)

    #For testing
    #schoolsInGeo2, schoolsInCoords, misteryLocations, misteryPoints = doTestSet(schoolsDictCollapsed,schoolsInGeo)

    candidatePoints, discrepancySchools = validatePoints(schoolsInGeo,offset,R,vs)

    #For testing
    #candidatePoints.append((8.486793733888566,-13.23540144248957))
    #candidatePoints.append((8.606129799837882,-10.589481768105736))

    finalPoints = collapsePoints(candidatePoints,minimal_spacing)

    i = 1
    print("Id,Lat,Lon,Collapsed")
    
    for p in finalPoints:
        lat, lon = p
        print(str(i) + ',' + str(lat) + ',' + str(lon)+",Yes")
        i += 1

if __name__ == "__main__":
    main()
