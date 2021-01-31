import numpy as np
from typing import List,Any,Tuple
import matplotlib.pyplot as plt



def main():
    size = 1000
    # buy_coordinates = np.random.randint(0,100,(size,2))
    # sell_coordinates = np.random.randint(0,100,(size,2))
    buy_coordinates = np.random.random((size,2))
    mask = np.repeat(True,len(buy_coordinates)//3)
    mask = np.append(mask,np.repeat(False,len(buy_coordinates)-len(buy_coordinates)//3))
    buy_coordinates[mask] = np.add(buy_coordinates[mask],5)

    mask = np.repeat(False,len(buy_coordinates)//3)
    mask = np.append(mask,np.repeat(True,len(buy_coordinates)//3))
    mask = np.append(mask,np.repeat(False,len(buy_coordinates)-len(buy_coordinates)//3*2))
    buy_coordinates[mask] = np.add(buy_coordinates[mask],3)
    
    sell_coordinates = np.random.random((size,2))
    # population = [{"id":str(i),"bpoint":list(b),"spoint":list(s)} for i,b,s in zip(range(size),buy_coordinates,sell_coordinates)]
    population = [{"id":str(i),"bpoint":b,"spoint":s} for i,b,s in zip(range(size),buy_coordinates,sell_coordinates)]

    buy_k,sell_k = find_kmeans_k(population)
    print(buy_k,sell_k)

    for p in population:
        print(p)

    print('')

    population = kmeans_clustering(population,buy_k,sell_k)
    for p in population:
        print(p)
    allcolors = list(cnames.values())
    buys = [[*p["bpoint"],allcolors[p["buy_cluster"]]] for p in population]
    sells = [[*p["spoint"],allcolors[p["sell_cluster"]]] for p in population]

    plt.scatter([b[0] for b in buys],[b[1] for b in buys],c=[b[2] for b in buys])
    # plt.scatter([s[0] for s in sells],[s[1] for s in sells],c=[s[2] for s in sells])
    plt.show()
    


def kmeans_clustering(population:List,buy_k:int,sell_k:int)->List:
    # Choose initial centroids indexes
    did_change = True
    centroids = np.random.randint(0,len(population),size=buy_k)
    buy_centroids = [population[c]["bpoint"] for c in centroids]

    while did_change:
        for i in range(len(population)):
            min_distance = (0,np.inf)
            for idx,point in enumerate(buy_centroids):
                dist = kdistance(population[i]["bpoint"],point)
                if dist < min_distance[1]:
                    min_distance = (idx,dist)

            if "buy_cluster" in population[i]:
                if population[i]["buy_cluster"] == min_distance[0]:
                   did_change = False
                else:
                    population[i]["buy_cluster"] = min_distance[0]
            else:
                population[i]["buy_cluster"] = min_distance[0]
    
        new_buy_centroids = []
        for c in range(buy_k):
            buy_cluster = [p["bpoint"] for p in population if p["buy_cluster"] == c]
            if len(buy_cluster) == 0:
                continue
            
            new_buy_centroid = []
            for d in range(len(buy_cluster[0])):
                new_buy_centroid.append(np.median([b[d] for b in buy_cluster]))
            new_buy_centroids.append(new_buy_centroid)

        buy_centroids = new_buy_centroids


    # Choose initial centroids indexes
    did_change = True
    centroids = np.random.randint(0,len(population),size=buy_k)
    sell_centroids = [population[c]["spoint"] for c in centroids]

    while did_change:
        for i in range(len(population)):
            min_distance = (0,np.inf)
            for idx,point in enumerate(sell_centroids):
                dist = kdistance(population[i]["spoint"],point)
                if dist < min_distance[1]:
                    min_distance = (idx,dist)
                    
            if "sell_cluster" in population[i]:
                if population[i]["sell_cluster"] == min_distance[0]:
                    did_change = False
                else:
                    population[i]["sell_cluster"] = min_distance[0]
            else:
                population[i]["sell_cluster"] = min_distance[0]

        new_sell_centroids = []
        for c in range(sell_k):
            sell_cluster = [p["spoint"] for p in population if p["sell_cluster"] == c]
            if len(sell_cluster) == 0:
                continue

            new_sell_centroid = []
            for d in range(len(sell_cluster[0])):
                new_sell_centroid.append(np.median([b[d] for b in sell_cluster]))
            new_sell_centroids.append(new_sell_centroid)
        
        sell_centroids = new_sell_centroids

    return population
 


def kdistance(kpointA:List[Any],kpointB:List[Any])->float:
    """Return a euclidian distance between 2 k dimensional points in space.""" 
    if len(kpointA) != len(kpointB):
        raise RuntimeError(
        "The dimensionality k-points must be the same, but encountered"\
        F" {kpointA} and {kpointB}")

    return np.sqrt(np.sum(np.power(np.subtract(kpointA,kpointB),2)))


def find_kmeans_k(population:List)->Tuple:
    """Calculate the k value for a kmeans cluster for this population. Returns
    both the buy and the sell values seperately.
    Returns a tuple of buy_k and sell_k"""
    memo = {}
    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                continue
            if F"{i}-{j}" in memo:
                continue
            elif F"{j}-{i}" in memo:
                continue
            else:
                buy = kdistance(population[i]["bpoint"],population[j]["bpoint"])
                sell = kdistance(population[i]["spoint"],population[j]["spoint"])
                memo[F"{i}-{j}"] = (buy,sell)

    all_distances = memo.values()
    all_buy_distances = np.array([t[0] for t in all_distances])
    all_sell_distances = np.array([t[1] for t in all_distances])
    
    buy_std = np.std(all_buy_distances)/4
    sell_std = np.std(all_sell_distances)/4
    
    buy_centers = np.arange(
        start=all_buy_distances.min(),
        stop=all_buy_distances.max(),
        step=buy_std)

    if (buy_centers[-1] + buy_std/2) < all_buy_distances.max():
        buy_centers = np.array([*buy_centers,buy_centers[-1]+buy_std])

    sell_centers = np.arange(
        start=all_sell_distances.min(),
        stop=all_sell_distances.max(),
        step=sell_std)

    if (sell_centers[-1] + sell_std/2) < all_sell_distances.max():
        sell_centers = np.array([*sell_centers,sell_centers[-1]+sell_std])

    return len(buy_centers),len(sell_centers)


def cluster(values,max_distance):
    clusters = []
    while len(values) > 0:
        seed = np.random.choice(values)
        
        cluster =[]
        for val in values:
            if abs(seed-val) <= max_distance:
                cluster.append(val)
        
        for val in cluster:
            values.remove(val)
        
        clusters.append(cluster)
    
    return clusters


def cluster_fast(values:np.array,max_distance):
    print("Distance: ",max_distance)
    clusters = []
    while values.shape[0] > 0:
        seed = np.random.choice(values)
        distances = np.abs(np.subtract(np.repeat(seed,values.shape[0]),values))

        # if 1 then we selected if 0 then we didnt
        mask = np.where(distances<max_distance,True,False)
        clusters.append(values[mask])
        values = values[~mask]

    return clusters


def cluster_fast_nonrandom(values:np.array,max_distance):
    print("Distance: ",max_distance)
    clusters = []
    centers = np.arange(values.min(),values.max(),max_distance)
    if (centers[-1] + max_distance/2) < values.max():
        centers = np.array([*centers,centers[-1]+max_distance])

    for center in centers:
        distances = np.abs(np.subtract(np.repeat(center,values.shape[0]),values))

        # if 1 then we selected if 0 then we didnt
        mask = np.where(distances<=max_distance/2,True,False)
        clusters.append(values[mask])

    if not np.array_equal(np.sort(values),np.sort(np.array([v for clus in clusters for v in clus]))):
        raise RuntimeError("The two arrays are different")

    if values.shape[0] != len([v for clus in clusters for v in clus]):
        print("Original values: ",values.shape[0])
        print("Clustered: ",len([v for clus in clusters for v in clus]))
        raise RuntimeError("Different lengths") 
    return clusters

cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

def speedtest():
    # this is bs
    l = np.random.randint(0,100,(100,1))

    from time import time

    start = time()
    memo = {}
    for i in range(l.shape[0]):
        for j in range(l.shape[0]):
            if i == j:
                continue
            if F"{i}-{j}" in memo:
                continue
            elif F"{j}-{i}" in memo:
                continue
            else:
                memo[F"{i}-{j}"] = kdistance([l[i]],[l[j]])
    delta = time() - start
    print("Memo loops Elapsed: ", delta) 

    start = time()
    matrix = np.cross(l,l)
    print(matrix)
    delta = time() - start
    print("Cross product: ", delta) 


def test_selected():
    l1 = [6,2,4,5,1]
    l2 = [0,2,4,3,6]

    for elA in l1:
        if len([t for t in l2 if t == elA]):
            continue
        else:
            for idxb,elB in enumerate(l2):
                if len([t for t in l1 if t == elB]):
                    continue
                else:
                    l2[idxb] = elA
                    break
                

    print(l1)
    print(l2)


if __name__ == "__main__":
    test_selected()
    # speedtest()
    # main()
