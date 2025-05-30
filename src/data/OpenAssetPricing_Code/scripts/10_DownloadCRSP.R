# Environment -------------------------------------------------------------
tic = Sys.time()

## LOGIN TO WRDS
user = getPass('wrds username: ')
pass = getPass('wrds password: ')

wrds <- dbConnect(Postgres(),
                  host='wrds-pgdata.wharton.upenn.edu',
                  port=9737,
                  dbname='wrds',
                  user=user,
                  password=pass,
                  sslmode='require')

numRowsToPull = -1 # Set to -1 for all rows and to some positive value for testing
yearmax_crspd = year(Sys.time()) # set to year(Sys.time()) for all years or 1930 or something for testing

# CRSP monthly ------------------------------------------------------------
# Follows in part: https://wrds-www.wharton.upenn.edu/pages/support/research-wrds/macros/wrds-macro-crspmerge/

m_crsp = dbSendQuery(conn = wrds, statement = 
                       "select a.permno, a.permco, a.date, a.ret, a.retx, a.vol, a.shrout, a.prc, a.cfacshr, a.bidlo, a.askhi,
                     b.comnam, b.shrcd, b.exchcd, b.siccd, b.ticker, b.shrcls,  -- from identifying info table
                     c.dlstcd, c.dlret                                -- from delistings table
                     from crsp.msf as a
                     left join crsp.msenames as b
                     on a.permno=b.permno
                     and b.namedt<=a.date
                     and a.date<=b.nameendt
                     left join crsp.msedelist as c
                     on a.permno=c.permno 
                     and date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
                     "
) %>% 
  # Pull data
  dbFetch(n = numRowsToPull) %>%
  as_tibble()

# write to disk 
write_fst(
  m_crsp 
  , paste0(pathThesisData,'raw/m_crsp_raw.fst')
)

Sys.time()
Sys.time() - tic

rm(m_crsp)
