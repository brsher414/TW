select distinct t2.segtype,t1.segdesc,t2.csegment from db_cate_segment t1
join db_dic_segment t2 
on t1.segno=t2.segno
and t1.catcode = t2.catcode
and t1.catcode = 'BEER'
AND t1.segcode not in ('CATEGORY','BRAND','SUBBRAND');
select * from db_dic_segment t1 where catcode = 'BEER';
