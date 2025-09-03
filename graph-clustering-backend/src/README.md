go mod edit -droprequire github.com/gilchrisn/graph-clustering-service     
go clean -modcache    

go get github.com/gilchrisn/graph-clustering-service@b8e7e9f28439c54829a6304868fb7d0936974164
go mod tidy
