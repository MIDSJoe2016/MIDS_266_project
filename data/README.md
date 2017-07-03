Processed source files are plain text with no markup.  (All text from unprocessed source files should end up here along with any raw text previously gathered.)

Current source for raw text source files include:
* State of the Union Address (http://stateoftheunionaddress.org/)
* The Collected Works of Abraham Lincoln (http://quod.lib.umich.edu/l/lincoln/)

Unprocessed source files include:
* **Presidential_News_Confs_(2009-).json**
(http://www.presidency.ucsb.edu/news_conferences.php)

Contains all news conferences with the President as speaker from 2009 onwards.  The `<speaker>` tag can be used to extract the contents of the `text` tag for a given President.
* **Presidential_Debates__(1960-).json**
(http://www.presidency.ucsb.edu/debates.php)

Contains all presidential candidate debates since 1960 including party debates (e.g., Republican nominee debates).  The `<date>` tag must first be to extract relevant years, and then checked for the speaker in question (e.g., TRUMP).
* **State_Union.json**
(http://www.presidency.ucsb.edu/sou.php)

Contains all State of the Union Addresses & Messages given since the early 1800's.  The `<date>` tag must is the sole indicator of the speaker in this file, but given that dates are so clearly known for these speeches, this is simple.  (e.g., Obama did 8 State of the Unions: 2009-2016 inclusive.)  Reference the ssource URL noted above for mapping of presidents to years.
