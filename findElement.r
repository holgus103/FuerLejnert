findElement <- function(list, length, val, counter)
{
  if(counter > length)
    return(FALSE)
  else if(list[[counter]] == val)
    return(TRUE)
  else
    findElement(list, length, val, counter+1)
}