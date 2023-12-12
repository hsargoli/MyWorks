

-- having 


-- 20  RANK 
SELECT bd.OINTCUSTID_ID, bd.OWNCUSTNAME,
rank() OVER (ORDER BY bd.PAN DESC) AS asas
FROM WCRDSHMA.BONCARD_DETAIL bd 


-- ...............................................................
-- 19 we want cust that exist more than one time in table
SELECT * from (
SELECT bd.OWNCUSTNAME ,count(*) AS tt FROM WCRDSHMA.BONCARD_DETAIL bd
GROUP BY bd.OWNCUSTNAME
) AS t 
WHERE tt > 1
WITH ur

SELECT bd.OWNCUSTNAME,bd.FIRSTCHARGE ,count(1) AS tt FROM WCRDSHMA.BONCARD_DETAIL bd
GROUP BY bd.OWNCUSTNAME
WITH ur

SELECT CUSTTYPE,BRANCH_ID FROM WRHSHMA.DIMCUSTIMPROVE d
fetch first 1000 rows only
WITH ur


-- ...............................................................
-- 18 multiple JOIN 
SELECT * FROM WCRDSHMA.DIMCARD d2 
INNER JOIN WRHSHMA.DIMCUSTIMPROVE d 
ON d2.BRANCH_ID = d.BRANCH_ID 
INNER JOIN WCRDSHMA.ACCESSCARD_DETAIL2 ad 
ON d2.PAN = ad.PAN 
WITH ur

-- ...............................................................
-- 17 DATE & between:  macna date in 1399
SELECT * FROM WCRDSHMA.CREDITCARD_DETAIL cd
WHERE cd.MACNADATE > 13991010
AND cd.MACNADATE < 14001010
WITH ur

SELECT * FROM WCRDSHMA.CREDITCARD_DETAIL cd
WHERE cd.MACNADATE BETWEEN 13991010 AND 14001010
WITH ur


-- ...............................................................
-- 17 CASE
SELECT cd.CNTRCTID,cd.REQUESTAMNT,cd.PAN, cd.MACNAAMOUNT,
CASE WHEN cd.MACNAAMOUNT = 0 THEN 'sefr'
ELSE 'gheyr sefr'
END  AS MACNAAMOUNT
FROM WCRDSHMA.CREDITCARD_DETAIL cd
WITH ur





-- ...............................................................
-- 16 group by - inner join - count 
SELECT d.REQUESTTYPE,d2.INCOMETYPE, COUNT(*) AS countt FROM WCRDSHMA.DIMCARDREQUEST d 
INNER JOIN WCRDSHMA.DIMCARDTYPE d2
ON d.CARDTYPE = d2.CARDTYPE 
GROUP BY d.REQUESTTYPE,d2.INCOMETYPE
FETCH FIRST 10 ROWS only
WITH ur


-- ...............................................................
-- 15 ineer JOIN 
/* we want join in two table
 * we must detect similar column in thats table
*/
SELECT d.CARDSTATUS,d.ISSUETYPE ,d2.CUSTTYPE, d2.COMPANYTYPE FROM WCRDSHMA.DIMCARD d 
INNER JOIN WRHSHMA.DIMCUSTIMPROVE d2 
ON d.BRANCH_ID = d2.BRANCH_ID 
WITH ur




-- # cross JOIN, dont want mention similar columns. it handle this
SELECT bd.OWNCUSTNAME ,bd.OWNINTCUSTID , cd.REQUESTAMNT FROM WCRDSHMA.BONCARD_DETAIL bd 
cross JOIN WCRDSHMA.CREDITCARD_DETAIL cd
WITH ur

SELECT bd.OWNCUSTNAME ,bd.OWNINTCUSTID , cd.REQUESTAMNT FROM WCRDSHMA.BONCARD_DETAIL bd 
FULL outer JOIN WCRDSHMA.CREDITCARD_DETAIL cd ON bd.PAN
WITH ur



-- ...............................................................
-- 14 group by
SELECT d2.CARDTYPE FROM WCRDSHMA.DIMCARDTYPE d2
GROUP BY d2.CARDTYPE  
WITH ur

SELECT d2.CARDTYPE,d2.CARDTYPEDESC FROM WCRDSHMA.DIMCARDTYPE d2
GROUP BY d2.CARDTYPE,d2.CARDTYPEDESC   
WITH ur


SELECT d2.CARDTYPE,count(*) FROM WCRDSHMA.DIMCARDTYPE d2
GROUP BY d2.CARDTYPE   
WITH ur



-- 13 aggregation function 
SELECT COUNT(d2.COMMISSIONFEE) AS co,
sum(d2.COMMISSIONFEE) AS su,
min(d2.COMMISSIONFEE) AS mi,
avg(d2.COMMISSIONFEE) AS av 
FROM WCRDSHMA.DIMCARDTYPE d2 
WITH ur


-- 12 operation on columns
SELECT d.ID, d.ISSUETYPE * d.CARDSTATUS FROM WCRDSHMA.DIMCARD d 
WITH ur

-- 11 concat two column 
SELECT d.ID, CONCAT (d.ISSUETYPE, d.REPLICA) AS conc , d.CARDSTATUS FROM WCRDSHMA.DIMCARD d 


-- 10 CONVERT ?

-- 9 order by == sort 
SELECT d.ID, d.ISSUETYPE, d.CARDSTATUS FROM WCRDSHMA.DIMCARD d 
ORDER BY d.ID desc


-- 8 from a sub query 
SELECT d.ID, d.ISSUETYPE, d.CARDSTATUS FROM WCRDSHMA.DIMCARD d 
WHERE d.ISSUETYPE IN (11,2,3)
WITH ur

-- 7 Like NOT practical in persian!!!
SELECT d.BRANCHTITLE, d.ZONETITLE FROM WRHSHMA.DIMBRANCH d
WHERE d.BRANCHTITLE LIKE '% %'

-- 6 Not in
SELECT d.ID, d.ISSUETYPE, d.CARDSTATUS FROM WCRDSHMA.DIMCARD d 
WHERE d.ISSUETYPE <> 0 and d.ISSUETYPE != 1


-- 5 condition with AND & OR
SELECT d.ID, d.ISSUETYPE, d.CARDSTATUS FROM WCRDSHMA.DIMCARD d 
WHERE d.ISSUETYPE = 0
AND d.id > 10 OR  d.ISSUETYPE = 1 
WITH ur

-- 4 simple condition
SELECT d.ID, d.ISSUETYPE FROM WCRDSHMA.DIMCARD d 
WHERE d.ID < 10
WITH ur

-- 3-1 DISTINCT 
SELECT  DISTINCT bd.OWNCUSTNAME FROM WCRDSHMA.BONCARD_DETAIL bd
WITH ur


-- 3 see certain number of result
SELECT d.ID, d.ISSUETYPE FROM WCRDSHMA.DIMCARD d 
FETCH FIRST 20 ROWS only
WITH ur


-- 2 see certain column
SELECT d.ID, d.ISSUETYPE FROM WCRDSHMA.DIMCARD d 
WITH ur




-- 1  see all column
SELECT * FROM WCRDSHMA.DIMCARD d 
WITH ur