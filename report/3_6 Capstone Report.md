**Gun Violence DSC Capstone Group Winter Quarter Report**  
Dylan Berry, Luis Martinez-Chun, Diego Osborn, Parth Shinde

**Overview**  
So far, we have gotten a basic understanding of the crime data. In addition to this, we took a step back. Instead of looking at purely gun deaths, we are instead looking at all homicides of any nature. We have plotted where each instance of a homicide occurred on a map, both with algorithmic clustering and as a scatterplot. We are currently only working with the Chicago Crimes dataset, as we felt no other city had data as well put together or as clean as Chicago’s.  
We are also working out the technical aspects of plotting the data in hexagons. By partitioning the city into uniform hex cells, we believe we can better isolate and analyze specific neighborhoods/areas.

**Data**  
Our main dataset is the dataset of all Chicago crime from 2001 to the present, but filtered for the primary type of crime to only include homicides. This effectively reduced the dataset from eight million rows originally to only fourteen thousand rows. The data consists of demographic information, the approximate location of the incident, and more details regarding the crime (first-degree murder, second-degree murder, etc.).   
We also have access to a few more supplemental datasets related to social infrastructure (Boeing, 2025\) and socioeconomic factors (Bureau, 2013\) to complement the crime data and provide helpful context. We want to continue working on integrating these more and mapping them onto their respective hex areas.

| Column Title | Data Type | Use |
| :---- | :---- | :---- |
| ID | Int | Id of the crime. Unique identifier. |
| Case number | String | Another type of identifier. |
| Date | String | When the crime occured |
| Block | String | Approximate location of incident |
| IUCR | Int | Illinois uniform crime reporting code, each number represents a different crime |
| Primary Type | String | Category of crime committed |
| Description | String | Specificity in regards to the crime committed, category of crime committed |
| Location Description | String | Where crime occured, categorically (street, house, parking lot etc) |
| Arrest | Boolean | Whether or not an arrest was made |
| Domestic | Boolean | Whether or not the incident was domestic or not |
| Beat | Int | Police station identifier |
| District | Int | District identifier |
| Ward | Float | Ward identifier |
| Community Area | Float | Community Area identifier |
| FBI Code | String | Identifier used by FBI to determine what crime was committed |
| X coordinate | Float | Approximate X coordinate of crime |
| Y coordinate | Float | Approximate Y coordinate of crime |
| Year | Int | What year crime occurred in |
| Updated On | String | When this piece of data was added |
| Latitude | Float | Approximate latitude of where crime was committed |
| Longitude | Float | Approximate longitude of where crime was committed |
| Location | Tuple | Column that combines latitude and longitude into one column with both values |

**Tools**  
Currently, we are using Pandas and NumPy to process the data and Folium to map it. We also plan to find more advanced libraries for the hexagonal map-making and other purposes as well.

**Assigned Tasks**  
Dylan Berry \- Worked on writing the report, managed the GitHub repository  
Luis Martinez-Chun \- Worked on finding corollary data  
Diego Osborn \- Worked on finding corollary data  
Parth Shinde \- Worked on plotting the data and grouping it with hexagons

**Resources**  
Boeing, G. (2025). Modeling and Analyzing Urban Networks and Amenities With OSMnx. Geographical Analysis. [https://doi.org/10.1111/gean.70009](https://doi.org/10.1111/gean.70009)  
Bureau, U. S. C. (2013, December 17). Selected socioeconomic indicators by neighborhood. Cityofchicago.org. [https://data.cityofchicago.org/Health-Human-Services/Selected-socioeconomic-indicators-by-neighborhood/i9hv-en6g/about\_data](https://data.cityofchicago.org/Health-Human-Services/Selected-socioeconomic-indicators-by-neighborhood/i9hv-en6g/about_data)  
Crimes \- 2001 to Present | City of Chicago | Data Portal. (n.d.). Data.cityofchicago.org. [https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about\_data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)  
Kim, D. (2019). Social determinants of health in relation to firearm-related homicides in the United States: A nationwide multilevel cross-sectional study. PLOS Medicine, 16(12), e1002978. [https://doi.org/10.1371/journal.pmed.1002978](https://doi.org/10.1371/journal.pmed.1002978)  
McCourt, Alex, Patel, Ayush, and Kagawa, Rose. Background Check and Licensing Policies for Firearm Purchase: Design, Implementation, and Enforcement Elements by State (1980-2019). Ann Arbor, MI: Inter-university Consortium for Political and Social Research \[distributor\], 2021-07-15. [https://doi.org/10.3886/E145221V1](https://doi.org/10.3886/E145221V1)