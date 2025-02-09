import React from "react"
import "../styles/model.css"
import { useLocation } from 'react-router-dom'
import Nav from 'react-bootstrap/Nav';
import { Link } from "react-router-dom";
import Navbar from 'react-bootstrap/Navbar';
import { Button } from "react-bootstrap";
function Model_Summary(){
    const location = useLocation()
    const tick  = location.state
    const ticker = tick["ticker_type"]
    
    console.log(ticker)
    return(
        <>
        <div  className = {`${ticker}_model main_details`} >
        <Navbar expand="xl" fixed="top" style={{paddingLeft:"15px", paddingTop:"20px"}}>
                <Nav>
                    <Link  to={'/home'}>
                    <Button style={{backgroundColor:"#4E6E4D", border:"none"}}>
                        Home
                    </Button>
                    </Link>
                </Nav>
        </Navbar>
        <div className="main_model_details"> 
            {ticker}
        </div>

        </div>
        </>

    )
}
export default Model_Summary