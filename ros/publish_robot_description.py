#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, rospy

def add_prefix_to_urdf(urdf_text: str, prefix: str) -> str:
    def repl_link(m): return f'{m.group(1)}="{prefix}{m.group(2)}"'
    text = re.sub(r'(<\s*link\s+name\s*=\s*")([^"]+)(")', lambda m: f'{m.group(1)}{prefix}{m.group(2)}{m.group(3)}', urdf_text)
    text = re.sub(r'(<\s*joint\s+name\s*=\s*")([^"]+)(")', lambda m: f'{m.group(1)}{prefix}{m.group(2)}{m.group(3)}', text)
    text = re.sub(r'(<\s*parent\s+link\s*=\s*")([^"]+)(")', repl_link, text)
    text = re.sub(r'(<\s*child\s+link\s*=\s*")([^"]+)(")', repl_link, text)
    return text

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", dest="urdf_path", type=str, required=True)
    p.add_argument("--prefixes", type=str, default="ref_,cmd_,eq_")
    args = p.parse_args()

    with open(args.urdf_path, "r") as f:
        urdf_text = f.read()

    rospy.init_node("publish_robot_description", anonymous=True)
    for pref in [s for s in args.prefixes.split(",") if s.strip()]:
        ns = pref[:-1] if pref.endswith("_") else pref
        param = f"/{ns}/robot_description"
        rospy.set_param(param, add_prefix_to_urdf(urdf_text, pref))
        rospy.loginfo("set %s (prefix='%s')", param, pref)

if __name__ == "__main__":
    main()
