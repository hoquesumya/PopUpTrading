import React from "react"
import NavbarHome from "./navbar";
import WathcListMain from "./watchlistMain";
import "../styles/watchlist.css"
function WatchList(){
    return(
        <div className ="watchlist">
             <NavbarHome></NavbarHome>
             <WathcListMain></WathcListMain>
        </div>
       
    )
}
export default WatchList