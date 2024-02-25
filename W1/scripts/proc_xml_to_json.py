import xml.etree.ElementTree as ET
import json
PATH_TO_XML = '../data/ai_challenge_s03_c010-full_annotation.xml'
PATH_TO_OUTPUT = '../output/'

frame_dict = {}

# Read the XML file
tree = ET.parse(PATH_TO_XML)

# Itarate over every track in annotations (root)
for track in tree.getroot():
    # Skip if it's not a track
    if track.tag != 'track':
        continue
    # Skipt if label is not car
    if track.attrib['label'] != 'car':
        continue
    # Get the id of the track
    track_id = track.attrib['id']
    print(f"Track ID: {track_id}")
    # Iterate over every frame in the track
    for box in track:
        # Get the frame number
        frame_number = box.attrib['frame']
        # Get the bounding box
        xtl = box.attrib['xtl']
        ytl = box.attrib['ytl']
        xbr = box.attrib['xbr']
        ybr = box.attrib['ybr']
        # Get outside, occuluded and keyframe
        outside = box.attrib['outside']
        occluded = box.attrib['occluded']
        keyframe = box.attrib['keyframe']
        # Get attribute inside the box
        for attr in box:
            # Get the content of the attribute
            attr_content = attr.text
        
        # print(f"Frame: {frame_number}, Bounding Box: ({xtl}, {ytl}), ({xbr}, {ybr}), Outside: {outside}, Occluded: {occluded}, Keyframe: {keyframe}, Is Parked: {is_parked}")
        # Add the frame to the dictionary
        if frame_number not in frame_dict:
            frame_dict[frame_number] = {}
        
        frame_dict[frame_number][track_id] = {
            'xtl': float(xtl),
            'ytl': float(ytl),
            'xbr': float(xbr),
            'ybr': float(ybr),
            'outside': int(outside),
            'occluded': int(occluded),
            'keyframe': int(keyframe),
            'is_parked': attr_content == 'true'
        }

# Sort the dictionary by frame number
frame_dict = dict(sorted(frame_dict.items(), key=lambda item: int(item[0])))
# Sort each frame dictionary by track id
for k, v in frame_dict.items():
    frame_dict[k] = dict(sorted(v.items(), key=lambda item: int(item[0])))

# Save the dictionary as a JSON file
with open(f"{PATH_TO_OUTPUT}frame_dict.json", 'w') as f:
    json.dump(frame_dict, f)