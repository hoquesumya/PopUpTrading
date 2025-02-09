import React from "react"
import { Row } from "react-bootstrap"
import { Button } from "react-bootstrap"
import "../styles/watchlistMain.css"
import { Link } from 'react-router-dom';
import NotificationsNoneIcon from '@mui/icons-material/NotificationsNone';
function WathcListMain(){
    var ticker_list = ["SPY", "VOO", "IVV", 'VTI', "QQQ"]

    return(
        <div className="ticker_list">
            <Row className="md-6 ticker_list_child align-items-stretch">
            {ticker_list.map((ticker, index) => (
                <Row key={index} className={`flex-fill border ticker-child-${index}`}
                    style ={{
                        display:"grid"
                    }}
                >
                    <div className ={`ticker-child-child${index}`}
                    style={{
                        margin:"auto",
                        display:"flex"
                    }}
                    >
                    <Link className="p-2 flex-grow-1 bd-highlight" 
                        style={{
                               
                            }}
                        to={'/model'} state={{ticker_type:ticker}}
                         
                        > <Button style={{fontWeight:900,
                            margin:"auto",
                            backgroundColor:"#4E6E4D",
                            border:"none",
                            textAlign:"left",
                            padding:"15px",
                            boxShadow:" 0 0 10px 4px rgba(255, 215, 0, 0.74)"
                            }}>
                            {ticker}
                        </Button></Link>
                        <Button className="p-3" style={{backgroundColor:"transparent",  marginRight: "10px", border: 
                            "2px solid #97C4BB"}}>
                            $0.7
                        </Button> 
                        <Button className="p-3" style={{backgroundColor:"transparent", border:"none"}}>
                            <NotificationsNoneIcon />
                        </Button>
                    </div>
                </Row>
            ))}
            </Row>
        </div>
    )
}
export default WathcListMain