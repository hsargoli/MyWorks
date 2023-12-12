-- ------------------------------------------------------------------------------------------------------
/* 
no of cards in each group of client
*/
-- ------------------------------------------------------------------------------------------------------

SELECT d.CUSTGROUP,COUNT(d2.ID)  FROM WRHSHMA.DIMCUSTIMPROVE d 
INNER JOIN WCRDSHMA.DIMCARD d2 ON d.BRANCH_ID = d2.BRANCH_ID 
GROUP BY d.CUSTGROUP 
WITH ur
-- ------------------------------------ ex of above query
SELECT d.CARDTYPEDESC ,COUNT(d2.ID) FROM WCRDSHMA.DIMCARDTYPE d 
INNER JOIN WCRDSHMA.DIMCARDREQUEST d2 
ON d.CARDTYPE = d2.CARDTYPE 
GROUP BY d.CARDTYPEDESC 
WITH ur
-- ------------------------------------ 




-- te'edad eshterak 2 jadval
SELECT COUNT(*) FROM WCRDSHMA.DIMCARD d 
INNER JOIN WRHSHMA.DIMCUSTIMPROVE d2 
ON d.BRANCH_ID = d2.BRANCH_ID 
WITH ur
-- 100898677


-- ..............................................................
-- ------------------------------------------------------------------------------------------------------
/*
 type of cust 
 */
-- ------------------------------------------------------------------------------------------------------

SELECT d1.CUSTTYPE, count(d2.ID) FROM WRHSHMA.DIMCUSTIMPROVE d1 
INNER JOIN WCRDSHMA.DIMCARD d2 
ON d1.BRANCH_ID = d2.BRANCH_ID 
GROUP BY d1.CUSTTYPE
WITH ur



-- type of customer
SELECT DISTINCT d.CUSTTYPE FROM WRHSHMA.DIMCUSTIMPROVE d 
WITH ur



SELECT d.CUSTTYPE,d.EDUCATION FROM WRHSHMA.DIMCUSTIMPROVE d 
WITH ur



-- ------------------------------------------------------------------------------------------------------
/*
 * branch cards
 */
-- ------------------------------------------------------------------------------------------------------
SELECT d2.BRANCH,count(d.id) FROM WCRDSHMA.DIMCARD d 
INNER JOIN WRHSHMA.DIMBRANCH d2 
ON d.BRANCH = d2.BRANCH 
WHERE d2.RECENTFLAG = 1
GROUP by d2.BRANCH
-- AND d2.BRANCH = 19364
WITH ur


SELECT d2.BRANCH,count(d.id) FROM WCRDSHMA.DIMCARD d 


-- ................................................
SELECT DISTINCT d.BRANCH FROM WRHSHMA.DIMBRANCH d 
WHERE d.RECENTFLAG = 1 
WITH ur

-- No of distinc branch where recent flag is 1
SELECT COUNT(DISTINCT d.BRANCH) FROM WRHSHMA.DIMBRANCH d 
WHERE d.RECENTFLAG = 1
WITH ur








-- ------------------------------------------------------------------------------------------------------
/* ********************************************************
 count of cards in each class
 */
-- ------------------------------------------------------------------------------------------------------
SELECT d2.CARDTYPE, d2.CARDTYPEDESC ,COUNT(1) FROM WCRDSHMA.DIMCARD d 
INNER JOIN WCRDSHMA.DIMCARDTYPE d2 
ON d.CARDTYPE_ID = d2.ID
GROUP BY d2.CARDTYPE, d2.CARDTYPEDESC
WITH ur
/* ----------------------------------------------------
-- XXX   
thats branch which end date > now    - endDate Vs expireDate
-- ----------------------------------------------------
SELECT DISTINCT d.BRANCH_ID,d.ENDDATE,d.ID FROM WCRDSHMA.DIMCARD d 
WHERE d.ENDDATE >14010901
WITH ur */



-- ----------------------------------------------------
SELECT count(DISTINCT d.ACCNO_ID) FROM WCRDSHMA.DIMCARD d 
WHERE d.ENDDATE > 14010901
WITH ur
-- result: 26252338


-- ----------------------------------------------------
SELECT d2.CARDTYPE, d2.CARDTYPEDESC, d.CARDTYPE_ID, COUNT(1) FROM WCRDSHMA.DIMCARD d 
INNER JOIN WCRDSHMA.DIMCARDTYPE d2 
ON d.CARDTYPE_ID = d2.ID
GROUP BY d2.CARDTYPE, d2.CARDTYPEDESC, d.CARDTYPE_ID
WITH ur

-- ----------------------------------------------------
SELECT DISTINCT CARDTYPE_ID FROM WCRDSHMA.DIMCARD d
WITH ur

-- ----------------------------------------------------
SELECT DISTINCT ID FROM WCRDSHMA.DIMCARDTYPE d 
WITH ur

SELECT DISTINCT CARDTYPE, CARDTYPEDESC, INCOMETYPE
FROM WCRDSHMA.DIMCARDTYPE WHERE INCOMETYPE IN (
SELECT INCOMETYPE FROM WCRDSHMA.DIMCARDTYPE
GROUP BY INCOMETYPE
HAVING count(DISTINCT CARDTYPE)>1
)



SELECT d.BRANCH, COUNT(d2.ID) FROM WRHSHMA.DIMBRANCH d 
INNER JOIN WCRDSHMA.DIMCARD d2 ON d.BRANCH = d2.BRANCH 
WHERE d.RECENTFLAG = 1
GROUP BY d.BRANCH 
WITH ur


-- ------------------------------------------------------------------------------------------------------
/*
 * 
 */
-- ------------------------------------------------------------------------------------------------------
 


SELECT DISTINCT STATUSX,STATUSXDESC,REQUESTTYPEDESC FROM WCRDSHMA.DIMCARDREQUEST d
WITH ur



SELECT d2.CARDTYPE, d2.CARDTYPEDESC ,COUNT(1) FROM WCRDSHMA.DIMCARD d 
INNER JOIN WCRDSHMA.DIMCARDTYPE d2 
ON d.CARDTYPE_ID = d2.ID
GROUP BY d2.CARDTYPE, d2.CARDTYPEDESC
WITH ur

SELECT d.ISSUETYPE,COUNT(*) FROM WCRDSHMA.DIMCARD d 
INNER JOIN WCRDSHMA.DIMCARDTYPE d2 
ON d.CARDTYPE_ID = d2.ID
WHERE d2.CARDTYPE = 6
GROUP BY d.ISSUETYPE 
WITH ur

SELECT DISTINCT BRANCHCARDSTATUS, COUNT(*) FROM WCRDSHMA.GIFTCARD_DETAIL gd
GROUP BY BRANCHCARDSTATUS 
with ur


SELECT DISTINCT d.COMMISSIONFEE, d.INCOMETYPE,d.CARDGROUPDESC FROM WCRDSHMA.DIMCARDTYPE d 
ORDER BY d.COMMISSIONFEE desc
WITH ur