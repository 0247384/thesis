import requests
from lxml import etree as ET
from data.road import Road, Segment, Node

api_url = 'http://api.openstreetmap.org/api/0.6/map'


def parse_osm_roads(root):
    node_elements = root.findall('node')
    nodes = {}

    for node_element in node_elements:
        id = node_element.attrib['id']
        lat = float(node_element.attrib['lat'])
        lon = float(node_element.attrib['lon'])
        nodes[id] = Node(lat, lon)

    way_elements = root.findall('way')
    accepted_values = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street',
                       'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']
    links = ['motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']
    road_segments = []
    roads = {}

    for way_element in way_elements:
        is_road = False
        is_public = True
        is_area = False
        name = None
        tags = way_element.findall('tag')
        for tag in tags:
            if 'k' in tag.attrib and 'v' in tag.attrib:
                key = tag.attrib['k']
                value = tag.attrib['v']
                if key == 'name':
                    name = value
                elif key == 'ref' and name is None:
                    name = value
                elif key == 'highway':
                    if value in accepted_values:
                        is_road = True
                        if value in links and name is None:
                            name = value
                elif key == 'access':
                    if value != 'yes':
                        is_public = False
                elif key == 'area':
                    if value == 'yes':
                        is_area = True
        if is_road and is_public and not is_area and name is not None:
            name = name.replace('_', '-').replace('/', '-')
            segment_nodes = []
            segment_nodes_xml = way_element.findall('nd')
            for node in segment_nodes_xml:
                id = node.attrib['ref']
                if id in nodes:
                    segment_nodes.append(nodes[id])
            segment = Segment(name, segment_nodes)
            road_segments.append(segment)

    print('Road segments found:', len(road_segments))

    for segment in road_segments:
        name = segment.name
        if name in roads:
            roads[name].segments.append(segment)
        else:
            roads[name] = Road(name, [segment])

    print('Roads found:', len(roads))

    return list(roads.values())


def get_roads_from_osm_api(bbox):
    payload = {'bbox': str(bbox['left']) + ',' + str(bbox['bottom']) + ',' + str(bbox['right']) + ',' + str(bbox['top'])}
    response = requests.get(api_url, params=payload)
    print('Status code OSM:', response.status_code)
    root = ET.fromstring(response.content)
    return parse_osm_roads(root)


def get_roads_from_osm_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return parse_osm_roads(root)


def get_roads_from_xml_file(path):
    roads = []
    tree = ET.parse(path)
    nb_segments = 0

    for road_element in tree.getroot():
        name = road_element.find('name').text
        segments = []

        for segment_element in road_element.findall('segment'):
            nb_segments += 1
            nodes = []

            for node_element in segment_element:
                lat = float(node_element.find('lat').text)
                lon = float(node_element.find('lon').text)
                nodes.append(Node(lat, lon))

            segments.append(Segment(name, nodes))

        roads.append(Road(name, segments))

    print('Road segments found:', nb_segments)
    print('Roads found:', len(roads))
    return roads


def save_roads_to_xml_file(roads, save_path):
    root = ET.Element('roads')

    for road in roads:
        road_element = ET.SubElement(root, 'road')
        ET.SubElement(road_element, 'name').text = road.name

        for segment in road.segments:
            segment_element = ET.SubElement(road_element, 'segment')

            for node in segment.nodes:
                node_element = ET.SubElement(segment_element, 'node')
                ET.SubElement(node_element, 'lat').text = repr(node.lat)
                ET.SubElement(node_element, 'lon').text = repr(node.lon)

    tree = ET.ElementTree(root)
    tree.write(save_path, pretty_print=True)
    print('Roads saved to', save_path)


if __name__ == '__main__':
    osm_path = '' # TODO
    xml_path = '' # TODO
    roads = get_roads_from_osm_file(osm_path)
    save_roads_to_xml_file(roads, xml_path)
