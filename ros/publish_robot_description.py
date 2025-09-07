#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import xml.etree.ElementTree as ET
import rospy

def add_prefix_tree(root: ET.Element, prefix: str) -> ET.Element:
    def pref(s: str) -> str:
        return s if s.startswith(prefix) else prefix + s

    # link name
    for link in root.findall(".//link"):
        name = link.get("name")
        if name:
            link.set("name", pref(name))

    # joint name + parent/child link
    for joint in root.findall(".//joint"):
        jn = joint.get("name")
        if jn:
            joint.set("name", pref(jn))
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and parent.get("link"):
            parent.set("link", pref(parent.get("link")))
        if child is not None and child.get("link"):
            child.set("link", pref(child.get("link")))
    return root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="path to original URDF (no prefix)")
    ap.add_argument("--prefixes", default="ref_,cmd_,eq_", help="comma-separated, e.g., ref_,cmd_,eq_")
    args = ap.parse_args()

    tree = ET.parse(args.urdf)
    base_root = tree.getroot()

    rospy.init_node("publish_robot_description_prefixed", anonymous=True)
    for p in [s for s in args.prefixes.split(",") if s.strip()]:
        # deep copy element tree
        root = ET.fromstring(ET.tostring(base_root, encoding="utf-8"))
        root = add_prefix_tree(root, p)
        xml_str = ET.tostring(root, encoding="unicode")
        ns = p[:-1] if p.endswith("_") else p
        param = f"/{ns}/robot_description"
        rospy.set_param(param, xml_str)
        rospy.loginfo("set %s (prefix='%s')", param, p)

if __name__ == "__main__":
    main()
