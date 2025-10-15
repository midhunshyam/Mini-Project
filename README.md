# Finding the Way a Home Faces (Simple Guide)

> A friendly explainer of the **GNAF Orientation Pipeline** — in plain language.



## What is this about?

Every home sits near a road. This little project figures out **which way each home faces toward its nearest road** (for example: North, South‑East, West, etc.). It also **double‑checks its own work** and writes a short report so you can trust the results.

You don’t need to be a programmer to understand the idea — this guide keeps things simple. 🙂



## Why would anyone want this?

- **Better maps & navigation:** Helps planners and councils understand street layouts and house frontage.
- **Sunlight & design:** Orientation hints at how much sunlight or shade a home may get.
- **Safety & services:** Emergency services and delivery routes benefit from knowing which way properties face.
- **Research & planning:** Useful for community studies, traffic flow, and urban development.



## What data does it use?

- A list of **addresses with locations** (latitude/longitude). In Australia, this often comes from the **GNAF** dataset.
- A map of **roads** (lines on a map).
- The script does **not** use personal information (like names). It only looks at **points on a map** and **where the roads are**.

> If you don’t have Australian data, that’s okay — the same idea works anywhere with addresses and roads.



## What does the program actually do? (High level)

1. **Load the maps**: It opens the addresses and roads.
2. **Pick nearby addresses**: It focuses on addresses that fall inside the overall roads area (faster and tidier).
3. **Find the closest road** to each address.
4. **Draw an invisible arrow** from the address to that road and measure the **direction** (bearing).
5. **Translate the direction** into one of **8 compass points**: N, NE, E, SE, S, SW, W, NW.
6. **Save the results** in a simple table (CSV file).
7. **Double‑check**: It recomputes the answers and compares them, producing a **quality report** so you can see how accurate it is.





## What do I get at the end?

You’ll see three files:

- **Results table** – Shows each address, the direction it faces, and the distance to the road.
- **Validation table** – A side‑by‑side comparison that confirms the results are sensible.
- **Run log** – A friendly report with stats (how many matched, any mismatches, and a “PASS” or “ATTENTION” summary).

These are plain text files (CSV and TXT), so you can open them with Excel, Google Sheets, or a basic text editor.



## How do I run it? (Short version)

Clone to this repository and follow the below instructions:

If you like, you can run the script from a terminal:

```bash
cd <path/to/Mini-Project/Scripts>
 
python Microburbs_MiniProject.py \
  --gnaf gnaf_prop.parquet \
  --roads roads.gpkg \
  --epsg_metric 7856 \
  --radii 30 60 120 250 500 \
  --results gnaf_orientation.csv \
  --qc gnaf_orientation_validated.csv \
  --log MiniProject_Output.txt
```

Don’t worry about the extra options — the defaults work for most cases. You’ll just need the addresses file and the roads file.

> Tip: If your area isn’t in Australia, swap `--epsg_metric` for a local metric map setting (your GIS friend can help).
> Note: The script doesn't create the repository structured directories. This is #TODO.


## A tiny bit of the “how” (no heavy math!)

- The program uses a standard map trick to measure **distance** accurately.
- It finds the **nearest point on the closest road** to each home.
- It then measures the **direction** of that nearest point from the home’s location.
- Finally, it rounds that direction into a simple compass label like **“NE”**.

This avoids guesswork and keeps the results consistent.



## Privacy & safety

- The script only uses **coordinates** and basic address strings, not personal details.
- It doesn’t connect to the internet or share data.
- Always check your local **data license** and **privacy rules** when using address datasets.



## FAQ

**Q: Do I need to be a GIS expert?**  
*A: Nope.* If you can run a small command and open a CSV file, you’re good. A GIS person can help you choose the right map settings for your region.

**Q: What if an address is far from any road?**  
*A: The program will say “no road nearby.”* You can increase the search distance if needed.

**Q: Are the results always perfect?**  
*A: No model is perfect.* That’s why the project **validates** itself and shows you a **match rate** and examples of any mismatches.

**Q: Can I use this outside Australia?**  
*A: Yes.* Just provide your own address list and road map files for your region.



## Example “story” of one address

1. The home is at latitude X, longitude Y.  
2. The nearest road is found a short distance away.  
3. Draw a line from the home to that road point — it points **South‑West**.  
4. The result says the home “faces **SW**,” distance 12.5 m, bearing 238.7°.  
5. The validation step runs again and confirms the same answer ! 



## Credits

- Built with free, open‑source tools (GeoPandas, Shapely, Pandas, PyArrow).  
- Address and road data come from public datasets — please follow their license rules.



## Conclusion

This project turns complex geospatial processing into **clear, friendly outputs**. Whether you’re a planner, a student, or just curious about your neighbourhood, it gives you **simple answers** to a surprisingly useful question: *“Which way does this place face?”*

