# Environmental Data Linkage Tool

## Background and Introduction

Environmental data refers to the comprehensive collection and integration of information representing an individual's or population's exposure to environmental, social, and lifestyle factors, including Social Determinants of Health (SDoH). These datasets are increasingly important for clinical research worldwide. However, because they are collected and maintained by different organizations, there is no centralized, unified database that allows researchers to quickly access, visualize, and download reliable data.

This project addresses the following needs:

1. A GIS-enabled database to store geospatial data  
2. Efficient data ingestion with minimal manual effort  
3. A lightweight web application for data cataloging, visualization, and basic analysis  
4. Local geocoding toolkits for converting addresses to coordinates or FIPS codes  

---

## Database Design

We use **PostgreSQL** with the **PostGIS** extension to support geospatial operations. Inspired by the [OHDSI GIS Working Group](https://github.com/OHDSI/GIS), our database includes core tables to store both geospatial and variable data.  

Key design elements:

* **`data_source` table:** Stores metadata from all source datasets, including SDoH, environmental data, and geometry data.  
* For each SDoH and environmental dataset, two tables are created:  
  - **`variable_index` table:** Captures attributes and properties of each variable  
  - **`variable_value` table:** Stores variable values linked to geocoded locations  
* Geospatial data is stored in dedicated tables and linked to variables using geoCodes such as FIPS, ZCTA, COUNTY, and STATE  

![Database Design](./assets/Database%20Design.jpeg)

---

## Toolkits

Most SDoH datasets use Census Tract (FIPS) boundaries. To support investigators, we provide **toolkits** to geocode source addresses into coordinates or FIPS codes.  

To protect patient privacy, geocoding should be executed **locally**. Our toolkit supports both:

- CSV file input  
- Direct extraction from OMOP CDM databases  

> If your institution already provides geocoding services, you may skip these toolkits.

### Documentation

Detailed instructions are available [here](https://github.com/bihorac-LAB/EnvironmentalData/blob/main/Tools/doc/UserManual.md).

### Container images

For exact reproducibility, use the images by digest. Docker accepts `name:tag@sha256:<digest>`, and the digest takes precedence over the tag.

| Step | Image | Digest |
|---|---|---|
| Geocoding | `prismaplab/exposome-geocoder:1.0.4` | `sha256:eaaec7646eaa7b6b445bdbbd2c055a15758181b206ebfdcf3159dd99cbe7d825` |
| Address matching (started by the geocoder) | `ghcr.io/degauss-org/geocoder:3.3.0` | `sha256:39bbcb4a2c8f832dbf3d957285bbe23081095a839ab162360044a7ddd539dbf4` |
| Linkage | `ghcr.io/chorus-ai/chorus-postgis-exposure:main` | `sha256:06768ff5163ce6c98031f98c7c368bf88a67795636e89236f966c68ccc1661ca` |

The linkage image is built by CHoRUS from [`postgis-exposure`](https://github.com/chorus-ai/chorus-container-apps/tree/894d540a72ef0913f5d83fb1543440280e5164e3/postgis-exposure) at revision `894d540a`, the same code as [`Tools/postgis-exposure`](Tools/postgis-exposure). Its `main` tag is rebuilt whenever that folder changes, so pin the digest rather than the tag.

### Demo Videos
- [FIPS Generation](https://www.loom.com/share/603607c1365342cb9fb2303566f529cc?sid=81cab3db-8189-4ad4-b1fd-baca7a73bf25)  
- [Environmental Linkage to Other Datasets](https://www.loom.com/share/bc4097b0d3db4f8f9132a06a49c17e71?sid=ad9671c1-6535-4bc9-b893-7e917efbcf75)
  
---

### Web Application

We developed a [web application](https://exposome.rc.ufl.edu/) that provides investigators with:

- A data catalog  
- The environmental data linkage tool  
- Data visualization features  

A basic data analysis tool will be included in the next release.  

The web application is deployed on [HiPerGator](https://www.rc.ufl.edu/about/hipergator/) PubApps.  

![External Investigator Workflow](./assets/External%20Investigator%20Workflow.png)

## Contributing to Environmental Data Linkage Tool

Thank you for your interest in contributing! To ensure smooth collaboration, please follow these guidelines:

### How to Contribute
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature/my-feature`).
5. Open a Pull Request describing your changes.

### Reporting Issues
- Use the **Issues** tab for bug reports, feature requests, or questions.
- Provide clear steps to reproduce any bugs and include any relevant screenshots or logs.

### Coding Standards
- Python scripts: PEP8 formatting
- Markdown: Use proper headings and code blocks
- Docker scripts: Ensure commands are tested and documented


## Contact / Questions

For questions, feedback, or collaboration requests regarding this repository, please reach out to:

**prismap lab** – [ic3-center@ufl.edu](mailto:ic3-center@ufl.edu)  

Or you can submit an issue directly in this repository, and we will respond promptly.

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
The linkage code in [`Tools/postgis-exposure`](Tools/postgis-exposure) comes from the CHoRUS project and keeps its own license, [`Tools/postgis-exposure/LICENSE`](Tools/postgis-exposure/LICENSE).
