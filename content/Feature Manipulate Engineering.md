---
title: 
draft: false
tags: 
date: 2024-08-15
---

## Read Web Data

- Define a transformer.
- Make HTTP calls using the HTTPCaller transformer.
- Turn a JSON response into features using the JSONFlattener transformer.


A **transformer** is an FME Workbench object that carries out feature restructuring.
**Schema** (sometimes known as "data model") can be described as the structure of a dataset, or, more accurately, a formal definition of a dataset’s structure.


### For Microsoft Excel data being read by FME Workbench, a feature type is:
- [ ] A Microsoft Excel XLS or XLSX file
- [x] A sheet in the Microsoft Excel file
- [ ] A row in the sheet
- [ ] A column in the sheet

### The Coordinate System parameter for the Microsoft Excel reader should be set if:
- [ ] The data will be viewed with a background map
- [ ] The data will be compared to other data in a different coordinate system
- [ ] The writer format requires a coordinate system
- [x] All of the above

### How many rows of data are in the BusinessOwners feature type?
- [ ] 56
- [x] 100
- [ ] 425
- [ ] 268

### While you create an FME workspace for your own use, you are an:
- [x] Author
- [ ] End-user
- [ ] FME Flow Developer
- [ ] Administrator

### In the Translation Log window, ERROR entries will be colored:
- [ ] Black
- [ ] Blue
- [ ] Green
- [x] Red

### The error in Sven's workspace was caused by:
- [ ] FME being unable to convert Microsoft Excel data to an Esri file geodatabase.
- [ ] Not connecting the correct feature types.
- [x] Not setting the Geometry parameter on the writer feature type.
- [ ] Connecting features with incompatible geometry types to the writer feature type.

### The Feature Type dialog box can be opened by:
- [ ] Double-clicking on the feature type on the canvas
- [ ] Clicking on the cog-wheel button on the feature type on the canvas
- [ ] Right-clicking on the feature type on the canvas and selecting "Properties" in the drop-down menu
- [x] All of the above

### Once you add writer feature types, you are free to connect them or delete them as you wish.
- [x] True
- [ ] False

### The name of a feature type always sets the name of the output file.
- [ ] True
- [x] False

### Navigating to the location of your written data requires remembering your file path and manually finding it using your operating system's file browser.
- [ ] True
- [x] False

### What is the size of the written geodatabase to the nearest KB? Choose the closest answer, as results will vary slightly.
- [x] 171
- [ ] 498
- [ ] 836
- [ ] 2,932

### After clicking on a port, holding down the CTRL key (⌘ or Shift on Mac) allows you to connect a feature connection to multiple ports.
- [x] True
- [ ] False

- **Who might benefit from your use of annotation and bookmarks as part of best practice/style?**

  - [ ] Other FME users in your organization
  - [ ] Customers to whom you deliver FME workspaces
  - [ ] Yourself in the future, coming back to edit the workspace
  - [x] All of the above

- **Bookmarks document sections of your workspace, while annotations document specific objects.**

  - [x] True
  - [ ] False

- **The default behaviour of the Visual Preview is to display the cache of the selected object on the canvas. This can be turned off by clicking which Visual Preview button?**

  - [ ] Display Control
  - [ ] Open in Data Inspector
  - [ ] Show/Hide Feature Information window
  - [x] Automatic Inspect on Selection

- **Visual Preview and Data Inspector are examples of a GIS.**

  - [ ] True
  - [x] False

- **What is the title of the northernmost artwork in the Public Art dataset?**

  - [ ] Welcome Figures
  - [x] Aerodynamic Forms in Space
  - [ ] Gate to the Northwest Passage
  - [ ] Flame of Peace

- **Right-clicking on column headings in Table view allows you to sort them:**

  - [ ] Natural
  - [ ] Alphabetical
  - [ ] Numeric
  - [x] All of the above

- **It is possible to show or hide columns from the Table view.**

  - [x] True
  - [ ] False

- **Additional attributes may be created by the writer if they are required by the format.**

  - [x] True
  - [ ] False

### How many points are in the point cloud?

- [ ] 1,054,031
- [ ] 491,384
- [ ] 5,458,240
- [x] 11,527,736

### What is the unit of the coordinate system used by the LAS file?

- [x] Meters
- [ ] Feet
- [ ] Decimal Degrees

### Feature Information shows:

- [ ] User defined attributes
- [ ] Format attributes
- [ ] List attributes
- [x] All of the above

### The 3D view in the Graphics View can be rotated using the:

- [ ] Zoom Extents tool
- [ ] Zoom In tool
- [ ] Pan tool
- [x] Orbit tool

### A template can contain:

- [ ] A workspace
- [ ] Source datasets
- [ ] Caches
- [x] All of the above

### Background maps, if selected, will display in both 2D and 3D views

- [ ] True
- [x] False

### What is the definition of a transformer?

- [x] An object in a translation that carries out feature restructuring.
- [ ] An object in a translation that describes an identifiable subset of records within a dataset.
- [ ] An object in a translation that writes to a destination dataset.
- [ ] An object in a translation that reads a source dataset.

### Which transformer is most commonly used for making API calls?

- [ ] Creator
- [x] HTTPCaller
- [ ] JSONExtractor
- [ ] JSONFragmenter

### When using the JSONFragmenter, what happens if you enable Flatten Query Result into Attributes but do not provide attribute names in the Attributes to Expose parameter?

- [ ] Performance will be negatively impacted
- [ ] The JSONFragmenter will output rejected features
- [ ] The translation will fail
- [x] Any flattened attributes will be unexposed

### Why do the points from the VertexCreator not appear in the correct place on the Visual Preview Graphics view?

- [ ] The Graphics view is not configured to show points
- [ ] The VertexCreator is not the correct transformer for creating points
- [x] They do not have a coordinate system set
- [ ] You have to manually move them by selecting them in Visual Preview and dragging them to the correct location.

### Your data appears to be in the incorrect place in the Graphics view, and the Feature Information Window reports the Coordinate System is Unknown. Which transformer do you need to use to fix this problem?

- [ ] CoordinateSystemExtractor
- [ ] CoordinateSystemRemover
- [x] CoordinateSystemSetter
- [ ] Reprojector

### Which of the following does not indicate your data is missing a coordinate system?

- [ ] The Feature Information Window reports that the features' Geometry > Coordinate System is set to Unknown.
- [ ] The Graphics view has a warning ribbon that says, "Some features in [port] may not align with the background map."
- [ ] The features do not appear where you expect.
- [ ] The Graphics view reports the data uses "Unknown Units" in the bottom right.
- [x] FME Workbench crashes when you try to inspect the data.

### One stream of features in your workspace uses the UTM83-10 coordinate system. Another stream uses BCALB-83. You would like to find where they intersect using the Intersector transformer. Which transformer do you need to use to ensure the datasets align properly before intersecting them?

- [ ] CoordinateSystemExtractor
- [ ] CoordinateSystemRemover
- [ ] CoordinateSystemSetter
- [x] Reprojector

![[Pasted image 20240823231444.png]]

### Where can you view the schema of your source data, i.e. the data you start with?

- [ ] The Transformer Gallery
- [ ] The writer entry in the Navigator
- [x] The reader feature type dialog
- [ ] The writer feature type dialog

### Opening the writer feature type dialog and adding a new attribute is a form of:

- [ ] Schema mapping
- [x] Schema editing
- [ ] Structure componentizing
- [ ] Data refreshing

### Which of the following operations is not possible using the AttributeManager?

- [ ] Renaming attributes
- [ ] Changing attribute types
- [ ] Changing attribute order
- [x] Sorting features by attribute values
- [ ] Setting attribute values

### What Attribute Definition mode would you use on a writer feature type if you wanted the schema to change to match incoming features?

- [x] Automatic
- [ ] Manual

### You just made a change to a transformer in the middle of your workspace and want to rewrite the output data to reflect the change. Which partial run button would you use after clicking on the transformer where you made changes?

- [ ] Run To This
- [x] Run From This
- [ ] Run Just This
- [ ] Run Between Selected

### To inspect a feature cache in Visual Preview, click the:

- [x] Green magnifying glass icon
- [ ] Transformer cogwheel icon
- [ ] Run button
- [ ] Canvas

### Partial runs can help speed up the authoring of workspaces by:

- [ ] Allowing you to only run part of your workspace.
- [ ] Allowing you to update a single cache.
- [ ] Allowing you to quickly inspect data you just added to the canvas.
- [ ] Allowing you to create a cache of web, database, or compressed data.
- [x] All of the above.

### You have a Tester reading road features. There are two tests: the first tests if the road type is a highway, the second tests if the road type is an off-ramp. Which Logic option do you need for the Tester to pass both highways and off-ramps features?

- [ ] AND
- [x] OR
- [ ] AND NOT
- [ ] OR NOT

### All FME workspaces must have multiple streams of data.

- [ ] True
- [x] False

### What happens when Automatic Inspect on Selection is enabled, and you click a transformer with multiple feature caches?

- [x] All feature caches are inspected
- [ ] No feature caches are inspected
- [ ] The top cache only is inspected
- [ ] The bottom cache only is inspected

### Which setting must be enabled to ask the user to optionally set a value for a user parameter at run time?

- [x] Published
- [ ] Required
- [ ] Disable Attribute Assignment
- [ ] Conditional Visibility

### FME Flow can run workspaces on a schedule.

- [x] True
- [ ] False

### User parameters let FME users:

- [x] Control how workspaces run
- [ ] Log in so the run event is tied to their user profile
- [ ] Control how fast their workspace runs
- [ ] Control who can run their workspace through Windows Active Directory

### Web connections let you share authentication information (logins, passwords, API keys, etc.) with other users of FME Flow without exposing your plaintext password.

- [x] True
- [ ] False

### Which of the following is not a valid way to access your workspace in FME Flow after publishing it?

- [x] From FME Workbench, click Run > Run on FME Flow.
- [ ] Login to FME Flow and click the link to the Run workspace page on the left sidebar.
- [ ] Click the Direct Link URL in the Translation Log after publishing the workspace from FME Workbench.

### The Run Workspace page lets users set published user parameters that control how the workspace will run.

- [x] True
- [ ] False

### If I want to view applicable results directly in my browser, which FME Flow service should I use when I run a workspace?

- [ ] Data Download
- [x] Data Streaming
- [ ] Job Submitter
- [ ] Notification

### From the exercise results, which Vancouver library branch has the fewest number of books available for borrowing (BookCount)?

- [ ] Accessible Services
- [ ] Central Branch
- [ ] Firehall
- [x] Strathcona

### After publishing a workspace to FME Flow, only the original author can run the workspace via the web interface.

- [ ] True
- [x] False

### If Fatima changes her mind and wants a weekly update instead of a monthly one, where does Frank need to go in the Automation to make the change?

- [ ] Email External Action > Email To parameter
- [ ] Run a workspace Action > Output Keys
- [ ] Schedule Trigger > Start parameter
- [x] Schedule Trigger > Recurrence parameter
- [ ] None of the above

### Which of the following workflows could be created using Automations?

- [ ] When an AutoCAD DWG file is uploaded to an Amazon S3 bucket, run it through a quality assurance testing workspace. If it passes, convert it to a GIS format and upload that to S3. If it fails, email the uploader.
- [ ] Automatically run a nightly database backup.
- [ ] When a field technician emails in a daily summary spreadsheet via email, email them back a report with a route map and weather forecast for their sites the next day.
- [ ] When a user adds an Excel file of addresses to a Dropbox folder, automatically add another file with all the addresses geocoded.
- [x] All of the above

### Which of the following scenarios would benefit from the use of an Automation running on a schedule?

- [ ] Automatically updating a database every time a file is added to an S3 bucket.
- [ ] Automatically providing a dataset whenever an email is received with a particular subject line.
- [ ] Automatically making an API call to an asset management system whenever a GIS layer is updated.
- [x] Automatically providing delivery drivers with a list and map of their daily deliveries every morning.

