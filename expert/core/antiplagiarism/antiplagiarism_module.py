from __future__ import annotations

import zeep
import asyncio
import datetime
from zeep.transports import AsyncTransport
import suds.client
import base64
import httpx
import time
import os

from app.libs.expert.expert.core.antiplagiarism.plagiarism_tools.schemas import SimpleCheckResult, Service, Source, Author, LoanBlock
from app.libs.expert.expert.core.antiplagiarism.plagiarism_tools.logger import logger


class AntiplagiatClient:
    """Module for detecting plagiarism in the source text using the Antiplagiat API.
    
    Args:
        login (str): Login of the registered user.
        password (str): Password of the registered user.
        company_name (str): Name of registered organization.
        apicorp_address (str, optional): Url of API access.
        antiplagiat_uri (str, optional): Url of company in "Antiplagiat" system.
    """
    def __init__(
        self,
        login: str,
        password: str,
        company_name: str,
        apicorp_address: str = "api.antiplagiat.ru:44902",
        antiplagiat_uri: str = "https://testapi.antiplagiat.ru"
    ) -> None:
        self.antiplagiat_uri = antiplagiat_uri
        self.login = login
        self.password = password
        self.company_name = company_name
        self.apicorp_address = apicorp_address
        self.client = suds.client.Client(f"https://{self.apicorp_address}/apiCorp/{self.company_name}?singleWsdl",
                                         username=self.login,
                                         password=self.password)
    
    def _get_doc_data(
        self,
        filename: str,
        external_user_id: str
    ):
        data = self.client.factory.create("DocData")
        data.Data = base64.b64encode(open(filename, "rb").read()).decode()
        data.FileName = os.path.splitext(filename)[0]
        data.FileType = os.path.splitext(filename)[1]
        data.ExternalUserID = external_user_id
        
        return data
    
    def simple_check(
        self,
        filename: str,
        author_surname: str = "",
        author_other_names: str = "",
        external_user_id: str = "ivanov",
        custom_id: str = "original"
    ) -> SimpleCheckResult:
        logger.info(f"SimpleCheck filename={filename}")
        
        data = self._get_doc_data(filename, external_user_id=external_user_id)
        
        docatr = self.client.factory.create("DocAttributes")
        personIds = self.client.factory.create("PersonIDs")
        personIds.CustomID = custom_id
        
        arr = self.client.factory.create("ArrayOfAuthorName")
        
        author = self.client.factory.create("AuthorName")
        author.OtherNames = author_other_names
        author.Surname = author_surname
        author.PersonIDs = personIds
        
        arr.AuthorName.append(author)
        docatr.DocumentDescription.Authors = arr
        
        # Downloading a file.
        try:
            uploadResult = self.client.service.UploadDocument(data, docatr)
        except Exception:
            raise
        
        # Document ID. If the downloaded file is not an archive, then
        # the list of downloaded documents will consist of one element.
        id = uploadResult.Uploaded[0].Id
        
        try:
            # Submit for verification using all search engines connected to the company.
            self.client.service.CheckDocument(id)
        # Submit for verification using only the native and the "wikipedia" search module.
        # See the get_tariff_info() example for getting a list of search modules.
        # >>> client.service.CheckDocument(id, ["wikipedia", COMPANY_NAME])
        except suds.WebFault:
            raise
        
        # Get the current status of the last check.
        status = self.client.service.GetCheckStatus(id)
        
        # Waiting cycle for the end of the check.
        while status.Status == "InProgress":
            time.sleep(status.EstimatedWaitTime * 0.1)
            status = self.client.service.GetCheckStatus(id)
        
        # If the check failed.
        if status.Status == "Failed":
            logger.error(f"An error occurred while validating the document {filename}: {status.FailDetails}")
        
        # Get a short report.
        report = self.client.service.GetReportView(id)
        
        logger.info(f"Report Summary: {report.Summary.Score:.2f}%")
        result = SimpleCheckResult(filename=os.path.basename(filename),
                                   plagiarism=f"{report.Summary.Score:.2f}%",
                                   services=[],
                                   author=Author())
        
        for checkService in report.CheckServiceResults:
            # Information for each search module.
            service = Service(service_name=checkService.CheckServiceName,
                              originality=f"{checkService.ScoreByReport.Legal:.2f}%",
                              plagiarism=f"{checkService.ScoreByReport.Plagiarism:.2f}%",
                              source=[])
            
            logger.info(f"Check service: {checkService.CheckServiceName}, "
                        f"Score.White={checkService.ScoreByReport.Legal:.2f}% "
                        f"Score.Black={checkService.ScoreByReport.Plagiarism:.2f}%")
            if not hasattr(checkService, "Sources"):
                result.services.append(service)
                continue
            for source in checkService.Sources:
                _source = Source(hash=source.SrcHash,
                                 score_by_report=f"{source.ScoreByReport:.2f}%",
                                 score_by_source=f"{source.ScoreBySource:.2f}%",
                                 name=source.Name,
                                 author=source.Author,
                                 url=source.Url)
                
                service.source.append(_source)
                # Information for each found source.
                logger.info(
                    f"\t{source.SrcHash}: Score={source.ScoreByReport:.2f}%({source.ScoreBySource:.2f}%), "
                    f'Name="{source.Name}" Author="{source.Author}"'
                    f' Url="{source.Url}"')
            
            # Get a full report.
            result.services.append(service)
        
        options = self.client.factory.create("ReportViewOptions")
        options.FullReport = True
        options.NeedText = True
        options.NeedStats = True
        options.NeedAttributes = True
        fullreport = self.client.service.GetReportView(id, options)
        
        logger.info(f"Author Surname={fullreport.Attributes.DocumentDescription.Authors.AuthorName[0].Surname} "
                    f"OtherNames={fullreport.Attributes.DocumentDescription.Authors.AuthorName[0].OtherNames} "
                    f"CustomID={fullreport.Attributes.DocumentDescription.Authors.AuthorName[0].PersonIDs.CustomID}")
        
        result.author.surname = fullreport.Attributes.DocumentDescription.Authors.AuthorName[0].Surname
        result.author.othernames = fullreport.Attributes.DocumentDescription.Authors.AuthorName[0].OtherNames
        result.author.custom_id = fullreport.Attributes.DocumentDescription.Authors.AuthorName[0].PersonIDs.CustomID
        
        loan_blocks = []
        if fullreport.Details.CiteBlocks:
            for block in fullreport.Details.CiteBlocks:
                loan_block = LoanBlock(text=fullreport.Details.Text[block.Offset:block.Offset + block.Length],
                                       offset=block.Offset,
                                       length=block.Length)
                loan_blocks.append(loan_block)
        result.loan_blocks = loan_blocks
        
        return result.dict()
    
    def _get_report_name(self, id, reportOptions):
        author = u""
        
        if reportOptions is not None:
            if reportOptions.Author:
                author = "_" + reportOptions.Author
        
        curDate = datetime.datetime.today().strftime("%Y%m%d")
        
        return f"Certificate_{id.Id}_{curDate}_{author}.pdf"
    
    def get_verification_report_pdf(
        self,
        filename: str,
        author: str,
        department: str,
        type: str,
        verifier: str,
        work: str,
        path: str | None = None,
        external_user_id: str = "ivanov"
    ):
        logger.info(f"Get report pdf: {filename}")
        
        data = self._get_doc_data(filename, external_user_id=external_user_id)
        
        uploadResult = self.client.service.UploadDocument(data)
        
        id = uploadResult.Uploaded[0].Id
        
        self.client.service.CheckDocument(id)
        
        status = self.client.service.GetCheckStatus(id)
        
        while status.Status == "InProgress":
            time.sleep(status.EstimatedWaitTime)
            status = self.client.service.GetCheckStatus(id)
        
        if status.Status == "Failed":
            logger.error(f"An error occurred while validating the document {filename}: {status.FailDetails}")
            return
        
        try:
            reportOptions = self.client.factory.create("VerificationReportOptions")
            reportOptions.Author = author # Full name of the author of the work.
            reportOptions.Department = department # Faculty (department).
            reportOptions.ShortReport = True # If a link to the summary required (QR code).
            reportOptions.Type = type # Type of the work.
            reportOptions.Verifier = verifier # Full name of the inspector.
            reportOptions.Work = work # Title of the work.
            
            reportWithFields = self.client.service.GetVerificationReport(id, reportOptions)
            
            decoded = base64.b64decode(reportWithFields)
            fileName = self._get_report_name(id, reportOptions)
            
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                filepath = os.path.join(path, f"{fileName}")
            else:
                filepath = fileName
            
            f = open(f"{filepath}", "wb")
            f.write(decoded)
        except suds.WebFault as e:
            if e.fault.faultcode == "a:InvalidArgumentException":
                raise Exception(
                    u"The document does not have a report/closed report, or None is passed as 'id' in GetVerificationReport: " + e.fault.faultstring)
            if e.fault.faultcode == "a:DocumentIdException":
                raise Exception(u"Specified invalid 'DocumentId'" + e.fault.faultstring)
            raise
        logger.info(f"Success create report in path: {filepath}")


class AsyncAntiplagiatClient:
    """Module for asynchronous detecting plagiarism in the source text using the Antiplagiat API.
    
    Args:
        login (str): Login of the registered user.
        password (str): Password of the registered user.
        company_name (str): Name of registered organization.
        apicorp_address (str, optional): Url of API access.
        antiplagiat_uri (str, optional): Url of company in "Antiplagiat" system.
    """
    def __init__(
        self,
        login: str,
        password: str,
        company_name: str,
        apicorp_address: str = "api.antiplagiat.ru:44902",
        antiplagiat_uri: str = "https://testapi.antiplagiat.ru"
    ):
        self.antiplagiat_uri = antiplagiat_uri
        self.login = login
        self.password = password
        self.company_name = company_name
        self.apicorp_address = apicorp_address
        self.httpx_client = httpx.AsyncClient(auth=(self.login, self.password))
        self.client = zeep.AsyncClient(
            f"https://{self.apicorp_address}/apiCorp/{self.company_name}?singleWsdl",
            transport=AsyncTransport(client=self.httpx_client))
        self.factory = self.client.type_factory("ns0")
    
    async def _get_doc_data(self, filename: str, external_user_id: str):
        Data = base64.b64encode(open(filename, "rb").read()).decode()
        FileName = os.path.splitext(filename)[0]
        FileType = os.path.splitext(filename)[1]
        ExternalUserID = external_user_id
        
        data = self.factory.DocData(Data=Data, FileName=FileName, FileType=FileType, ExternalUserID=ExternalUserID)
        
        return data
    
    async def simple_check(
        self,
        filename: str,
        author_surname: str = "",
        author_other_names: str = "",
        external_user_id: str = "ivanov",
        custom_id: str = "original"
    ) -> SimpleCheckResult:
        logger.info(f"SimpleCheck filename={filename}")
        
        data = await self._get_doc_data(filename, external_user_id=external_user_id)
        docatr = self.factory.DocAttributes()
        personIds = self.factory.PersonIDs()
        personIds.CustomID = personIds
        arr = self.factory.ArrayOfAuthorName()
        author = self.factory.AuthorName()
        author.OtherNames = author_other_names
        author.Surname = author_surname
        author.PersonIDs = personIds
        arr.AuthorName.append(author)
        # docatr.DocumentDescription.Authors = arr
        
        try:
            uploadResult = await self.client.service.UploadDocument(data, docatr)
        except Exception:
            raise
        
        id = uploadResult[0]["Id"]
        
        try:
            await self.client.service.CheckDocument(id)
        except suds.WebFault:
            raise
        
        status = await self.client.service.GetCheckStatus(id)
        
        while status.Status == "InProgress":
            await asyncio.sleep(status.EstimatedWaitTime * 0.1)
            status = await self.client.service.GetCheckStatus(id)
        
        if status.Status == "Failed":
            print(f"An error occurred while validating the document {filename}: {status.FailDetails}")
        
        report = await self.client.service.GetReportView(id)
        
        logger.info(f"Report Summary: {report.Summary.Score:.2f}%")
        result = SimpleCheckResult(filename=os.path.basename(filename),
                                   plagiarism=f"{report.Summary.Score:.2f}%",
                                   services=[],
                                   author=Author())
        
        for checkService in report.CheckServiceResults:
            # Information for each search module.
            service = Service(service_name=checkService.CheckServiceName,
                              originality=f'{checkService.ScoreByReport.Legal:.2f}%',
                              plagiarism=f'{checkService.ScoreByReport.Plagiarism:.2f}%',
                              source=[])

            logger.info(f"Check service: {checkService.CheckServiceName}, "
                        f"Score.White={checkService.ScoreByReport.Legal:.2f}% "
                        f"Score.Black={checkService.ScoreByReport.Plagiarism:.2f}%")
            if not hasattr(checkService, "Sources"):
                result.services.append(service)
                continue
            for source in checkService.Sources:
                _source = Source(hash=source.SrcHash,
                                 score_by_report=f'{source.ScoreByReport:.2f}%',
                                 score_by_source=f'{source.ScoreBySource:.2f}%',
                                 name=source.Name,
                                 author=source.Author,
                                 url=source.Url)
                
                service.source.append(_source)
                # Information for each found source.
                logger.info(
                    f'\t{source.SrcHash}: Score={source.ScoreByReport:.2f}%({source.ScoreBySource:.2f}%), '
                    f'Name="{source.Name}" Author="{source.Author}"'
                    f' Url="{source.Url}"')
            
            # Get a full report.
            result.services.append(service)
        
        options = self.factory.ReportViewOptions()
        options.FullReport = True
        options.NeedText = True
        options.NeedStats = True
        options.NeedAttributes = True
        fullreport = await self.client.service.GetReportView(id, options)
        
        # Authors are not filled in because it is not possible to correctly send the request to the server.
        result.author.surname = None
        result.author.othernames = None
        result.author.custom_id = None
        
        loan_blocks = []
        if fullreport.Details.CiteBlocks:
            for block in fullreport.Details.CiteBlocks:
                loan_block = LoanBlock(text=fullreport.Details.Text[block.Offset:block.Offset + block.Length],
                                       offset=block.Offset,
                                       length=block.Length)
                loan_blocks.append(loan_block)
        result.loan_blocks = loan_blocks
        
        return result.dict()
    
    async def _get_report_name(self, id, reportOptions):
        author = u""
        
        if reportOptions is not None:
            if reportOptions.Author:
                author = "_" + reportOptions.Author
        
        curDate = datetime.datetime.today().strftime("%Y%m%d")
        
        return f"Certificate_{id.Id}_{curDate}_{author}.pdf"

    async def get_verification_report_pdf(
        self,
        filename: str,
        author: str,
        department: str,
        type: str,
        verifier: str,
        work: str,
        path: str | None = None,
        external_user_id: str = "ivanov"
    ):
        logger.info("Get report pdf:" + filename)

        data = await self._get_doc_data(filename, external_user_id=external_user_id)

        uploadResult = await self.client.service.UploadDocument(data)

        id = uploadResult[0]["Id"]

        await self.client.service.CheckDocument(id)

        status = await self.client.service.GetCheckStatus(id)

        while status.Status == "InProgress":
            await asyncio.sleep(status.EstimatedWaitTime * 0.1)
            status = await self.client.service.GetCheckStatus(id)

        if status.Status == "Failed":
            logger.error(f"An error occurred while validating the document {filename}: {status.FailDetails}")
            return
        
        try:
            reportOptions = self.factory.VerificationReportOptions()
            
            reportOptions.Author = author # Full name of the author of the work.
            reportOptions.Department = department # Faculty (department).
            reportOptions.ShortReport = True # If a link to the summary required (QR code).
            reportOptions.Type = type # Type of the work.
            reportOptions.Verifier = verifier # Full name of the inspector.
            reportOptions.Work = work # Title of the work.
            
            reportWithFields = await self.client.service.GetVerificationReport(id, reportOptions)
            # No decoding needed.
            # decoded = base64.b64decode(reportWithFields)
            
            fileName = await self._get_report_name(id, reportOptions)
            
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                filepath = os.path.join(path, f"{fileName}")
            else:
                filepath = fileName
            
            f = open(f"{filepath}", "wb")
            f.write(reportWithFields)
            logger.info(f"Success create report in path: {filepath}")
        except Exception as exc:
            logger.error(f"Error: {exc}")