#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division


def distance_euclidean(instance1, instance2):
    """Computes the distance between two instances. Instances should be tuples of equal length.
    Returns: Euclidean distance
    Signature: ((attr_1_1, attr_1_2, ...), (attr_2_1, attr_2_2, ...)) -> float"""

    def detect_value_type(attribute):
        """Detects the value type (number or non-number).
        Returns: (value type, value casted as detected type)
        Signature: value -> (str or float type, str or float value)"""
        from numbers import Number
        attribute_type = None
        if isinstance(attribute, Number):
            attribute_type = float
            attribute = float(attribute)
        else:
            attribute_type = str
            attribute = str(attribute)
        return attribute_type, attribute

    # check if instances are of same length
    if len(instance1) != len(instance2):
        raise AttributeError("Instances have different number of arguments.")
    # init differences vector
    differences = [0] * len(instance1)
    # compute difference for each attribute and store it to differences vector
    for i, (attr1, attr2) in enumerate(zip(instance1, instance2)):
        type1, attr1 = detect_value_type(attr1)
        type2, attr2 = detect_value_type(attr2)
        # raise error is attributes are not of same data type.
        if type1 != type2:
            raise AttributeError("Instances have different data types.")
        if type1 is float:
            # compute difference for float
            differences[i] = attr1 - attr2
        else:
            # compute difference for string
            if attr1 == attr2:
                differences[i] = 0
            else:
                differences[i] = 1
    # compute RMSE (root mean squared error)
    rmse = (sum(map(lambda x: x ** 2, differences)) / len(differences)) ** 0.5
    return rmse


class LOF:
    """Helper class for performing LOF computations and instances normalization."""

    def __init__(self, instances, normalize=True, distance_function=distance_euclidean):
        self.instances = instances
        self.normalize = normalize
        self.distance_function = distance_function
        if normalize:
            self.normalize_instances()

    def compute_instance_attribute_bounds(self):
        min_values = [float("inf")] * len(self.instances[0])  # n.ones(len(self.instances[0])) * n.inf
        max_values = [float("-inf")] * len(self.instances[0])  # n.ones(len(self.instances[0])) * -1 * n.inf
        for instance in self.instances:
            min_values = tuple(map(lambda x, y: min(x, y), min_values, instance))  # n.minimum(min_values, instance)
            max_values = tuple(map(lambda x, y: max(x, y), max_values, instance))  # n.maximum(max_values, instance)
        self.max_attribute_values = max_values
        self.min_attribute_values = min_values

    def normalize_instances(self):
        """Normalizes the instances and stores the infromation for rescaling new instances."""
        if not hasattr(self, "max_attribute_values"):
            self.compute_instance_attribute_bounds()
        new_instances = []
        for instance in self.instances:
            new_instances.append(
                self.normalize_instance(instance))  # (instance - min_values) / (max_values - min_values)
        self.instances = new_instances

    def normalize_instance(self, instance):
        return tuple(map(lambda value, max, min: (value - min) / (max - min) if max - min > 0 else 0,
                         instance, self.max_attribute_values, self.min_attribute_values))

    def local_outlier_factor(self, min_pts, instance):
        """The (local) outlier factor of instance captures the degree to which we call instance an outlier.
        min_pts is a parameter that is specifying a minimum number of instances to consider for computing LOF value.
        Returns: local outlier factor
        Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
        if self.normalize:
            instance = self.normalize_instance(instance)
        return local_outlier_factor(min_pts, instance, self.instances, distance_function=self.distance_function)


def k_distance(k, instance, instances, distance_function=distance_euclidean):
    # TODO: implement caching
    """Computes the k-distance of instance as defined in paper. It also gatheres the set of k-distance neighbours.
    Returns: (k-distance, k-distance neighbours)
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> (float, ((attr_j_1, ...),(attr_k_1, ...), ...))"""
    distances = {}
    for instance2 in instances:
        distance_value = distance_function(instance, instance2)
        if distance_value in distances:
            distances[distance_value].append(instance2)
        else:
            distances[distance_value] = [instance2]
    distances = sorted(distances.items())
    neighbours = []
    k_sero = 0
    k_dist = None
    for dist in distances:
        k_sero += len(dist[1])
        neighbours.extend(dist[1])
        k_dist = dist[0]
        if k_sero >= k:
            break
    return k_dist, neighbours


def reachability_distance(k, instance1, instance2, instances, distance_function=distance_euclidean):
    """The reachability distance of instance1 with respect to instance2.
    Returns: reachability distance
    Signature: (int, (attr_1_1, ...),(attr_2_1, ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(k, instance2, instances, distance_function=distance_function)
    return max([k_distance_value, distance_function(instance1, instance2)])


def local_reachability_density(min_pts, instance, instances, **kwargs):
    """Local reachability density of instance is the inverse of the average reachability
    distance based on the min_pts-nearest neighbors of instance.
    Returns: local reachability density
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    reachability_distances_array = [0] * len(neighbours)  # n.zeros(len(neighbours))
    for i, neighbour in enumerate(neighbours):
        reachability_distances_array[i] = reachability_distance(min_pts, instance, neighbour, instances, **kwargs)
    sum_reach_dist = sum(reachability_distances_array)
    if sum_reach_dist == 0:
        return float('inf')
    return len(neighbours) / sum_reach_dist


def local_outlier_factor(min_pts, instance, instances, **kwargs):
    """The (local) outlier factor of instance captures the degree to which we call instance an outlier.
    min_pts is a parameter that is specifying a minimum number of instances to consider for computing LOF value.
    Returns: local outlier factor
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    instance_lrd = local_reachability_density(min_pts, instance, instances, **kwargs)
    lrd_ratios_array = [0] * len(neighbours)
    for i, neighbour in enumerate(neighbours):
        instances_without_instance = set(instances)
        instances_without_instance.discard(neighbour)
        neighbour_lrd = local_reachability_density(min_pts, neighbour, instances_without_instance, **kwargs)
        lrd_ratios_array[i] = neighbour_lrd / instance_lrd
    return sum(lrd_ratios_array) / len(neighbours)


def outliers(k, instances, **kwargs):
    """Simple procedure to identify outliers in the dataset."""
    instances_value_backup = instances
    outliers = []
    for i, instance in enumerate(instances_value_backup):
        instances = list(instances_value_backup)
        instances.remove(instance)
        l = LOF(instances, **kwargs)
        value = l.local_outlier_factor(k, instance)
        if value > 1:
            outliers.append({"lof": value, "instance": instance, "exp_index": i})
    outliers.sort(key=lambda o: o["lof"], reverse=True)
    return outliers


# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/7 15:24
# @Author  : gsunwu@163.com
# @File    : source.py
# @Description:

if __name__ == '__main__':
    # -*- coding: utf8 -*-
    instances = [
        (-4.8447532242074978, -5.6869538132901658),
        (1.7265577109364076, -2.5446963280374302),
        (-1.9885982441038819, 1.705719643962865),
        (-1.999050026772494, -4.0367551415711844),
        (-2.0550860126898964, -3.6247409893236426),
        (-1.4456945632547327, -3.7669258809535102),
        (-4.6676062022635554, 1.4925324371089148),
        (-3.6526420667796877, -3.5582661345085662),
        (6.4551493172954029, -0.45434966683144573),
        (-0.56730591589443669, -5.5859532963153349),
        (-5.1400897823762239, -1.3359248994019064),
        (5.2586932439960243, 0.032431285797532586),
        (6.3610915734502838, -0.99059648246991894),
        (-0.31086913190231447, -2.8352818694180644),
        (1.2288582719783967, -1.1362795178325829),
        (-0.17986204466346614, -0.32813130288006365),
        (2.2532002509929216, -0.5142311840491649),
        (-0.75397166138399296, 2.2465141276038754),
        (1.9382517648161239, -1.7276112460593251),
        (1.6809250808549676, -2.3433636210337503),
        (0.68466572523884783, 1.4374914487477481),
        (2.0032364431791514, -2.9191062023123635),
        (-1.7565895138024741, 0.96995712544043267),
        (3.3809644295064505, 6.7497121359292684),
        (-4.2764152718650896, 5.6551328734397766),
        (-3.6347215445083019, -0.85149861984875741),
        (-5.6249411288060385, -3.9251965527768755),
        (4.6033708001912093, 1.3375110154658127),
        (-0.685421751407983, -0.73115552984211407),
        (-2.3744241805625044, 1.3443896265777866)]

    lof = outliers(5, instances)

    for outlier in lof:
        print(outlier["lof"], outlier["instance"])

    from matplotlib import pyplot as p

    x, y = zip(*instances)
    p.scatter(x, y, 20, color="#0000FF")

    for outlier in lof:
        value = outlier["lof"]
        instance = outlier["instance"]
        color = "#FF0000" if value > 1 else "#00FF00"
        p.scatter(instance[0], instance[1], color=color, s=(value - 1) ** 2 * 10 + 20)

    p.show()
